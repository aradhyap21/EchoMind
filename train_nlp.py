# ── SETUP VERIFICATION ────────────────────────────────────────────────────────
# Run this ONCE before training:
#
#   1. Create virtual environment:
#      python -m venv echomind-env
#      echomind-env\Scripts\activate        (Windows)
#      source echomind-env/bin/activate     (Linux/Mac)
#
#   2. Install PyTorch with CUDA 12.1 (for RTX 4050):
#      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#
#   3. Install other dependencies:
#      pip install transformers==4.44.0 datasets==2.20.0 evaluate==0.4.2
#      pip install accelerate==0.33.0 scikit-learn==1.5.1
#      pip install matplotlib==3.9.0 seaborn==0.13.2 huggingface_hub==0.24.5
#
#   4. Set your HuggingFace token as environment variable:
#      set HF_TOKEN=your_token_here        (Windows)
#      export HF_TOKEN=your_token_here     (Linux/Mac)
#      Get token from: https://huggingface.co/settings/tokens (Write access)
#
#   5. Run:
#      python train_nlp.py
#
# Expected training time on RTX 4050: 60-90 minutes for 5 epochs
#
# ── WINDOWS MULTIPROCESSING NOTE ─────────────────────────────────────────────
# Windows uses the "spawn" start method for child processes (unlike Linux/Mac
# which use "fork"). When PyTorch DataLoader spawns worker processes with
# num_workers > 0, each worker re-imports this entire module. Without an
# if __name__ == "__main__" guard the training code re-executes in every worker,
# triggering "bootstrapping phase" RuntimeError before training even starts.
#
# Fix applied here:
#   - All training logic lives inside main()
#   - multiprocessing.freeze_support() called first (required for frozen exes)
#   - dataloader_num_workers=0  → single-process loading, no spawn, zero risk
#     (on Windows with an SSD the difference vs num_workers=4 is negligible;
#      GPU is the bottleneck, not the dataloader)
# ─────────────────────────────────────────────────────────────────────────────

# ── SECTION 0 — IMPORTS ───────────────────────────────────────────────────────
# Imports stay at top-level — this is safe. Only *execution* must be guarded.
import os
import json
import time
import random
import shutil
import warnings
import multiprocessing
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader  # noqa: F401 (kept for completeness)

from datasets import load_dataset, DatasetDict, Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,  # noqa: F401
    pipeline,
)
import evaluate
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from huggingface_hub import login

warnings.filterwarnings("ignore", category=UserWarning)

# ── CONSTANTS — module-level is fine; they carry no execution side-effects ────
MODEL_NAME = "roberta-base"

OUTPUT_DIR     = "./echomind-nlp-output"
MODEL_DIR      = f"{OUTPUT_DIR}/model"
LOGS_DIR       = f"{OUTPUT_DIR}/logs"
HUB_MODEL_NAME = "echomind-emotion-model"

MAX_LENGTH      = 256
BATCH_SIZE      = 32
GRAD_ACCUM      = 2
LEARNING_RATE   = 1e-5
NUM_EPOCHS      = 5
WARMUP_RATIO    = 0.06
WEIGHT_DECAY    = 0.01
LABEL_SMOOTHING = 0.1
SEED            = 42

ID2LABEL = {0:"joy", 1:"sadness", 2:"anger", 3:"fear", 4:"surprise", 5:"disgust", 6:"neutral"}
LABEL2ID   = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

GOEMOTIONS_MAPPING = {
    "joy":"joy","amusement":"joy","excitement":"joy","admiration":"joy",
    "approval":"joy","gratitude":"joy","love":"joy","optimism":"joy",
    "pride":"joy","relief":"joy","caring":"joy",
    "sadness":"sadness","grief":"sadness","disappointment":"sadness",
    "remorse":"sadness","embarrassment":"sadness",
    "anger":"anger","annoyance":"anger","disapproval":"anger",
    "fear":"fear","nervousness":"fear",
    "surprise":"surprise","confusion":"surprise","curiosity":"surprise","realization":"surprise",
    "disgust":"disgust",
    "neutral":"neutral","desire":"neutral",
}

EMOTION_COLORS = {
    "joy":"#E8A030","sadness":"#6B9FFF","anger":"#D94040",
    "fear":"#A78BFA","surprise":"#00BFA0","disgust":"#86EFAC","neutral":"#8A8278",
}


# ── CLASS DEFINITIONS — safe at module level (no side-effects on import) ──────

class EchoMindTrainer(Trainer):
    """
    Custom Trainer with:
    1. Class-weighted cross-entropy loss — corrects GoEmotions class imbalance.
    2. Label smoothing — prevents overconfidence, improves F1 on rare classes.

    Label smoothing: at ε=0.1 the correct label target is 0.9 not 1.0,
    which stops the model from being overconfident and improves calibration.
    """

    def __init__(self, class_weights, label_smoothing=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights   = class_weights
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    # ── SECTION 1 — REPRODUCIBILITY + BANNER ─────────────────────────────────
    print(f"\n{'='*60}\n SECTION 1 — CONSTANTS AND CONFIG\n{'='*60}\n")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

    print(f"Model:            {MODEL_NAME}")
    print(f"Batch size:       {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
    print(f"Learning rate:    {LEARNING_RATE}")
    print(f"Epochs:           {NUM_EPOCHS}")
    print(f"Max seq length:   {MAX_LENGTH}")
    print(f"Label smoothing:  {LABEL_SMOOTHING}")
    print(f"Seed:             {SEED}")

    # ── SECTION 2 — GPU VERIFICATION ─────────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 2 — GPU VERIFICATION\n{'='*60}\n")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available.")
        print("Install PyTorch with CUDA support:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return

    device     = torch.device("cuda")
    gpu_name   = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"GPU:   {gpu_name}")
    print(f"VRAM:  {gpu_memory:.1f} GB")

    if gpu_memory < 5.0:
        print(f"\nWARNING: Only {gpu_memory:.1f}GB VRAM. Reduce BATCH_SIZE to 16.")

    for d in [OUTPUT_DIR, MODEL_DIR, LOGS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {Path(OUTPUT_DIR).resolve()}")
    print(f"Training start:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── SECTION 3 — LABEL DEFINITIONS ────────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 3 — LABEL DEFINITIONS\n{'='*60}\n")
    print(f"Label mapping: {len(GOEMOTIONS_MAPPING)} GoEmotions labels → {NUM_LABELS} classes")
    for label in ID2LABEL.values():
        mapped = [k for k, v in GOEMOTIONS_MAPPING.items() if v == label]
        print(f"  {label:<12} ← {', '.join(mapped)}")

    # ── SECTION 4 — LOAD DATASET ─────────────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 4 — LOAD GOEMOTIONS DATASET\n{'='*60}\n")
    print("Downloading GoEmotions from HuggingFace Hub...")
    raw_dataset     = load_dataset("google-research-datasets/go_emotions", "simplified")
    ORIGINAL_LABELS = raw_dataset["train"].features["labels"].feature.names
    print(f"\nRaw dataset:")
    for split, data in raw_dataset.items():
        print(f"  {split:<12}: {len(data):>6,} examples")
    print(f"Original label count: {len(ORIGINAL_LABELS)}")

    # ── SECTION 5 — PROCESS AND REMAP LABELS ─────────────────────────────────
    print(f"\n{'='*60}\n SECTION 5 — PROCESS AND REMAP LABELS\n{'='*60}\n")

    def process_example(example):
        if len(example["labels"]) == 0:
            return None
        primary_id   = example["labels"][0]
        primary_name = ORIGINAL_LABELS[primary_id]
        mapped       = GOEMOTIONS_MAPPING.get(primary_name)
        if mapped is None:
            return None
        return {"text": example["text"].strip(), "label": LABEL2ID[mapped], "label_name": mapped}

    processed = {}
    for split in ["train", "validation", "test"]:
        results, skipped = [], 0
        for ex in raw_dataset[split]:
            out = process_example(ex)
            if out:
                results.append(out)
            else:
                skipped += 1
        processed[split] = results
        print(f"  {split:<12}: {len(results):>6,} kept  ({skipped} skipped)")

    dataset = DatasetDict({s: HFDataset.from_list(d) for s, d in processed.items()})

    print(f"\nClass distribution (train set):")
    dist  = Counter(dataset["train"]["label_name"])
    total = len(dataset["train"])
    for emotion in ID2LABEL.values():
        count = dist.get(emotion, 0)
        bar   = "█" * int(count / total * 40)
        print(f"  {emotion:<12} {count:>5,}  {count/total*100:>5.1f}%  {bar}")

    # ── SECTION 6 — CLASS WEIGHTS ─────────────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 6 — CLASS WEIGHTS\n{'='*60}\n")
    train_labels         = np.array(dataset["train"]["label"])
    class_weights        = compute_class_weight(
        class_weight="balanced", classes=np.arange(NUM_LABELS), y=train_labels,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights (higher = rarer class):")
    for i, w in enumerate(class_weights):
        bar = "▓" * int(w * 6)
        print(f"  {ID2LABEL[i]:<12}  {w:.4f}  {bar}")

    # ── SECTION 7 — TOKENIZER AND TOKENIZATION ───────────────────────────────
    print(f"\n{'='*60}\n SECTION 7 — TOKENIZER AND TOKENIZATION\n{'='*60}\n")
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Vocabulary size:   {tokenizer.vocab_size:,}")
    print(f"Using MAX_LENGTH:  {MAX_LENGTH}")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    print("\nTokenizing dataset...")
    t0        = time.time()
    tokenized = dataset.map(
        tokenize, batched=True, batch_size=256,
        remove_columns=["text", "label_name"],
        desc="Tokenizing", num_proc=1,   # num_proc=1 — safe on Windows
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    print(f"Tokenization complete in {time.time() - t0:.1f}s")

    sample_lengths = [len(tokenized["train"][i]["input_ids"]) for i in range(500)]
    print(f"\nToken length stats (sample 500):")
    print(f"  Mean:   {np.mean(sample_lengths):.1f}")
    print(f"  Median: {np.median(sample_lengths):.1f}")
    print(f"  95th %: {np.percentile(sample_lengths, 95):.1f}")
    print(f"  Max:    {max(sample_lengths)}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── SECTION 8 — MODEL ─────────────────────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 8 — MODEL\n{'='*60}\n")
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model on:         {next(model.parameters()).device}")
    print(f"Total params:     {total_params:,}  ({total_params/1e6:.1f}M)")
    print(f"Trainable params: {trainable_params:,}  ({trainable_params/1e6:.1f}M)")

    # ── SECTION 9 — METRICS ───────────────────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 9 — METRICS\n{'='*60}\n")
    f1_metric  = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds          = np.argmax(logits, axis=-1)
        f1_weighted    = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        accuracy       = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        per_class      = f1_metric.compute(
            predictions=preds, references=labels,
            average=None, labels=list(range(NUM_LABELS))
        )["f1"]
        out = {"f1_weighted": round(f1_weighted, 4), "accuracy": round(accuracy, 4)}
        for i, f1 in enumerate(per_class):
            out[f"f1_{ID2LABEL[i]}"] = round(f1, 4)
        return out

    print("Metrics: f1_weighted, accuracy + per-class F1 for all 7 emotions.")

    # ── SECTION 10 — TRAINING ARGUMENTS ──────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 10 — TRAINING ARGUMENTS\n{'='*60}\n")
    steps_per_epoch = len(tokenized["train"]) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = int(total_steps * WARMUP_RATIO)

    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps:     {total_steps:,}")
    print(f"  Warmup steps:    {warmup_steps}")
    print(f"  dataloader_num_workers: 0  (Windows spawn-safe setting)")

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=2,
        logging_dir=LOGS_DIR,
        logging_steps=100,
        report_to="none",
        fp16=True,
        fp16_full_eval=True,
        # ── WINDOWS FIX ───────────────────────────────────────────────────────
        # dataloader_num_workers=0 disables worker processes entirely.
        # On Windows, num_workers > 0 spawns child processes that re-import
        # this module, causing the "bootstrapping phase" RuntimeError.
        # With num_workers=0 data loading runs in the main process — safe and
        # still fast because the GPU (not the dataloader) is the bottleneck.
        dataloader_num_workers=0,
        dataloader_pin_memory=True,   # Still useful with num_workers=0 on CUDA.
        seed=SEED,
    )

    # ── SECTION 11 — INITIALIZE TRAINER AND TRAIN ────────────────────────────
    print(f"\n{'='*60}\n SECTION 11 — TRAIN\n{'='*60}\n")
    trainer = EchoMindTrainer(
        class_weights=class_weights_tensor,
        label_smoothing=LABEL_SMOOTHING,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    print(f"Estimated time on RTX 4050: 60–90 minutes")
    print("=" * 60)

    t_start      = time.time()
    train_result = trainer.train()
    t_end        = time.time()

    duration_min = (t_end - t_start) / 60
    print("=" * 60)
    print(f"Training complete.  Duration: {duration_min:.1f} min")
    print(f"Train loss:   {train_result.metrics['train_loss']:.4f}")
    print(f"Steps/sec:    {train_result.metrics['train_steps_per_second']:.2f}")

    # ── SECTION 12 — TRAINING CURVES ─────────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 12 — TRAINING CURVES\n{'='*60}\n")
    log_history  = trainer.state.log_history
    train_logs   = [x for x in log_history if "loss" in x and "eval_loss" not in x]
    eval_logs    = [x for x in log_history if "eval_loss" in x]
    train_steps  = [x["step"]                  for x in train_logs]
    train_losses = [x["loss"]                  for x in train_logs]
    train_lrs    = [x.get("learning_rate", 0)  for x in train_logs]
    eval_epochs  = [x["epoch"]                 for x in eval_logs]
    eval_losses  = [x["eval_loss"]             for x in eval_logs]
    eval_f1s     = [x["eval_f1_weighted"]      for x in eval_logs]
    eval_accs    = [x["eval_accuracy"]         for x in eval_logs]
    per_class_history = {
        e: [x.get(f"eval_f1_{e}", 0) for x in eval_logs] for e in ID2LABEL.values()
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#111110")
    axes = axes.flatten()
    plots = [
        (axes[0], train_steps, train_losses, "#E8A030", "Training Loss",            "Step",  "Loss"),
        (axes[1], eval_epochs, eval_losses,  "#6B9FFF", "Validation Loss",          "Epoch", "Loss"),
        (axes[2], eval_epochs, eval_f1s,     "#00BFA0", "Validation F1 (weighted)", "Epoch", "F1"),
        (axes[3], eval_epochs, eval_accs,    "#A78BFA", "Validation Accuracy",      "Epoch", "Accuracy"),
        (axes[4], train_steps, train_lrs,    "#EDE5D0", "Learning Rate (cosine)",   "Step",  "LR"),
    ]
    for ax, x, y, color, title, xlabel, ylabel in plots:
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=3)
        ax.set_facecolor("#1C1B19")
        ax.set_title(title, color="#EDE5D0", fontsize=10, pad=8)
        ax.set_xlabel(xlabel, color="#8A8278", fontsize=9)
        ax.set_ylabel(ylabel, color="#8A8278", fontsize=9)
        ax.tick_params(colors="#8A8278", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2D2B27")
        ax.grid(True, color="#2D2B27", linewidth=0.5, alpha=0.5)

    ax6 = axes[5]
    ax6.set_facecolor("#1C1B19")
    for emotion, values in per_class_history.items():
        ax6.plot(eval_epochs, values, color=EMOTION_COLORS[emotion],
                 linewidth=1.5, label=emotion, marker="o", markersize=3)
    ax6.set_title("Per-Class F1 Over Epochs", color="#EDE5D0", fontsize=10, pad=8)
    ax6.set_xlabel("Epoch", color="#8A8278", fontsize=9)
    ax6.set_ylabel("F1", color="#8A8278", fontsize=9)
    ax6.tick_params(colors="#8A8278", labelsize=8)
    ax6.legend(fontsize=7, facecolor="#1C1B19", labelcolor="#EDE5D0",
               edgecolor="#2D2B27", loc="lower right")
    for spine in ax6.spines.values():
        spine.set_edgecolor("#2D2B27")
    ax6.grid(True, color="#2D2B27", linewidth=0.5, alpha=0.5)

    plt.suptitle("EchoMind — NLP Training Curves (roberta-base)",
                 color="#EDE5D0", fontsize=13, y=1.01)
    plt.tight_layout()
    curves_path = f"{OUTPUT_DIR}/training_curves.png"
    plt.savefig(curves_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {curves_path}")
    if eval_f1s:
        best_epoch = eval_f1s.index(max(eval_f1s)) + 1
        print(f"Best validation F1: {max(eval_f1s):.4f}  (epoch {best_epoch})")

    # ── SECTION 13 — TEST SET EVALUATION ─────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 13 — TEST SET EVALUATION\n{'='*60}\n")
    print("Evaluating on held-out test set...")
    test_results = trainer.evaluate(tokenized["test"])
    f1_val  = test_results["eval_f1_weighted"]
    print(f"\n=== TEST SET RESULTS ===")
    print(f"  Loss:          {test_results['eval_loss']:.4f}")
    print(f"  F1 (weighted): {f1_val:.4f}  {'PASS' if f1_val >= 0.65 else 'BELOW TARGET (0.65)'}")
    print(f"  Accuracy:      {test_results['eval_accuracy']:.4f}")
    print("\n  Per-class F1:")
    for emotion in ID2LABEL.values():
        val = test_results.get(f"eval_f1_{emotion}", 0)
        bar = "█" * int(val * 20)
        print(f"    {emotion:<12}  {val:.4f}  {bar}")

    # ── SECTION 14 — CONFUSION MATRIX ────────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 14 — CONFUSION MATRIX\n{'='*60}\n")
    pred_output = trainer.predict(tokenized["test"])
    preds       = np.argmax(pred_output.predictions, axis=-1)
    true_labels = pred_output.label_ids
    names       = list(ID2LABEL.values())
    cm_raw      = confusion_matrix(true_labels, preds)
    cm_norm     = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#111110")
    for ax, data, title, fmt in [
        (axes[0], cm_raw,  "Confusion Matrix — Counts",       "d"),
        (axes[1], cm_norm, "Confusion Matrix — Row Normalized", ".2f"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, xticklabels=names, yticklabels=names,
                    cmap="YlOrBr", ax=ax, linewidths=0.4, linecolor="#111110",
                    cbar_kws={"shrink": 0.75})
        ax.set_facecolor("#1C1B19")
        ax.set_title(title, color="#EDE5D0", fontsize=11, pad=10)
        ax.set_xlabel("Predicted", color="#8A8278")
        ax.set_ylabel("True Label", color="#8A8278")
        ax.tick_params(colors="#8A8278", labelsize=9)

    plt.suptitle("EchoMind — roberta-base Emotion Classifier",
                 color="#EDE5D0", fontsize=12, y=1.01)
    plt.tight_layout()
    cm_path = f"{OUTPUT_DIR}/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {cm_path}")

    report      = classification_report(true_labels, preds, target_names=names, digits=4)
    print("\n=== CLASSIFICATION REPORT ===")
    print(report)
    report_path = f"{OUTPUT_DIR}/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"EchoMind NLP — roberta-base\n")
        f.write(f"Trained: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(report)
    print(f"Saved: {report_path}")

    # ── SECTION 15 — SAVE MODEL AND METADATA ─────────────────────────────────
    print(f"\n{'='*60}\n SECTION 15 — SAVE MODEL AND METADATA\n{'='*60}\n")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    metadata = {
        "project": "EchoMind", "model_base": MODEL_NAME,
        "task": "emotion-classification", "num_labels": NUM_LABELS,
        "id2label": ID2LABEL, "label2id": LABEL2ID,
        "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM, "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS,
        "label_smoothing": LABEL_SMOOTHING, "scheduler": "cosine", "fp16": True,
        "dataset": "google-research-datasets/go_emotions (simplified)",
        "goemotions_mapping": GOEMOTIONS_MAPPING,
        "test_f1_weighted": round(test_results["eval_f1_weighted"], 4),
        "test_accuracy": round(test_results["eval_accuracy"], 4),
        "training_duration_min": round(duration_min, 1),
        "trained_on": gpu_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "windows_compat": {"dataloader_num_workers": 0, "main_guard": True},
    }
    meta_path = f"{OUTPUT_DIR}/echomind_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")

    for artifact in ["training_curves.png", "confusion_matrix.png", "classification_report.txt"]:
        src = f"{OUTPUT_DIR}/{artifact}"
        dst = f"{MODEL_DIR}/{artifact}"
        if Path(src).exists():
            shutil.copy(src, dst)

    print(f"\nModel directory contents:")
    for fi in sorted(Path(MODEL_DIR).iterdir()):
        print(f"  {fi.name:<45} {fi.stat().st_size/1024:>8.1f} KB")

    # ── SECTION 16 — INFERENCE TESTS ─────────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 16 — INFERENCE TESTS\n{'='*60}\n")
    emotion_pipeline = pipeline(
        task="text-classification",
        model=trainer.model,
        tokenizer=tokenizer,
        device=0,
        top_k=None,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    def predict_emotion(text: str) -> dict:
        """
        Inference — same return shape as JS mock predictText() in echomind-app.html.
        Drop-in replacement when wiring the real model into the Gradio frontend.
        """
        if not text or not text.strip():
            return {
                "emotions":   {e: (1.0 if e == "neutral" else 0.0) for e in ID2LABEL.values()},
                "dominant":   "neutral", "confidence": 1.0, "latency_ms": 0.0,
            }
        t0       = time.time()
        raw      = emotion_pipeline(text.strip())[0]
        ms       = (time.time() - t0) * 1000
        total    = sum(r["score"] for r in raw)
        emotions = {r["label"]: round(r["score"] / total, 4) for r in raw}
        dominant = max(emotions, key=emotions.get)
        return {
            "emotions": emotions, "dominant": dominant,
            "confidence": round(emotions[dominant], 4), "latency_ms": round(ms, 1),
        }

    TEST_SENTENCES = [
        ("I just got accepted into my dream university!",       "joy"),
        ("I haven't slept properly in weeks and I miss home.",  "sadness"),
        ("This is absolutely unacceptable behavior.",           "anger"),
        ("I have no idea what comes next and that scares me.",  "fear"),
        ("Wait — that was actually possible?",                  "surprise"),
        ("That is genuinely repulsive behavior.",               "disgust"),
        ("I went to the store and bought some groceries.",      "neutral"),
    ]

    print("\n=== INFERENCE TESTS ===\n")
    correct = 0
    for text, expected in TEST_SENTENCES:
        result  = predict_emotion(text)
        match   = "PASS" if result["dominant"] == expected else f"FAIL (expected {expected})"
        correct += (result["dominant"] == expected)
        print(f"Text:      {text[:65]}")
        print(f"Predicted: {result['dominant'].upper()} ({result['confidence']:.1%})  {match}")
        print(f"Latency:   {result['latency_ms']:.0f}ms")
        top3 = sorted(result["emotions"].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top 3:     " + "  |  ".join(f"{e} {v:.1%}" for e, v in top3))
        print()
    print(f"Inference test accuracy: {correct}/{len(TEST_SENTENCES)} ({correct/len(TEST_SENTENCES):.0%})")

    # ── SECTION 17 — PUSH TO HUGGINGFACE HUB ─────────────────────────────────
    print(f"\n{'='*60}\n SECTION 17 — PUSH TO HUGGINGFACE HUB\n{'='*60}\n")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN not set — skipping Hub push.")
        print("Set with:  set HF_TOKEN=your_token  (Windows)")
        print("Model saved locally at:", MODEL_DIR)
    else:
        login(token=hf_token)
        print(f"Pushing to Hub: {HUB_MODEL_NAME}")
        trainer.push_to_hub(HUB_MODEL_NAME, use_auth_token=hf_token)
        tokenizer.push_to_hub(HUB_MODEL_NAME, use_auth_token=hf_token)
        print(f"Model live at: https://huggingface.co/{HUB_MODEL_NAME}")

    # ── SECTION 18 — WRITE PREDICT.PY ────────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 18 — WRITE PREDICT.PY\n{'='*60}\n")
    predict_script = '''\
"""
EchoMind — NLP inference
Loads fine-tuned roberta-base from HuggingFace Hub.
Replace HUB_MODEL_ID with your actual model path.
"""
import time
from transformers import pipeline

HUB_MODEL_ID = "YOUR_USERNAME/echomind-emotion-model"
_pipe = None

def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "text-classification", model=HUB_MODEL_ID,
            top_k=None, truncation=True, max_length=256, device=0,
        )
    return _pipe

def predict_text(text: str) -> dict:
    """Same signature as JS mock predictText() in echomind-app.html."""
    if not text or not text.strip():
        emotions = {"joy":0,"sadness":0,"anger":0,"fear":0,"surprise":0,"disgust":0,"neutral":1.0}
        return {"emotions":emotions,"dominant":"neutral","confidence":1.0,"latency_ms":0.0}
    t0  = time.time()
    raw = get_pipeline()(text.strip())[0]
    ms  = (time.time() - t0) * 1000
    total    = sum(r["score"] for r in raw)
    emotions = {r["label"]: round(r["score"] / total, 4) for r in raw}
    dominant = max(emotions, key=emotions.get)
    return {"emotions":emotions,"dominant":dominant,
            "confidence":round(emotions[dominant],4),"latency_ms":round(ms,1)}

if __name__ == "__main__":
    print(predict_text("I just got the job offer!"))
    print(predict_text("I miss my old friends so much."))
'''
    predict_path = f"{OUTPUT_DIR}/predict.py"
    with open(predict_path, "w") as f:
        f.write(predict_script)
    print(f"Saved: {predict_path}")

    # ── SECTION 19 — FINAL SUMMARY ───────────────────────────────────────────
    print(f"\n{'='*60}\n SECTION 19 — FINAL SUMMARY\n{'='*60}\n")
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Model:         roberta-base (fine-tuned)")
    print(f"  Test F1:       {test_results['eval_f1_weighted']:.4f}  "
          f"{'(PASS)' if test_results['eval_f1_weighted'] >= 0.65 else '(BELOW TARGET 0.65)'}")
    print(f"  Test Accuracy: {test_results['eval_accuracy']:.4f}  "
          f"{'(PASS)' if test_results['eval_accuracy'] >= 0.63 else '(BELOW TARGET 0.63)'}")
    print(f"  Duration:      {duration_min:.1f} minutes")
    print(f"  GPU:           {gpu_name}")
    print(f"\n  Output files:")
    for fi in sorted(Path(OUTPUT_DIR).rglob("*")):
        if fi.is_file():
            rel = str(fi.relative_to(OUTPUT_DIR))
            print(f"    {rel:<50} {fi.stat().st_size/1024:>7.1f} KB")
    print(f"\n  Next: replace JS predictText() mock with predict.py")
    print("=" * 60)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
# CRITICAL for Windows:
#   freeze_support() — required if this is ever packaged as a frozen exe
#   if __name__ == "__main__" — prevents worker processes from re-executing main()
#   Without this guard, every DataLoader worker spawned by PyTorch re-imports
#   this module and triggers the RuntimeError at the "bootstrapping phase".
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
