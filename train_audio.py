# -*- coding: utf-8 -*-
# ── SETUP ─────────────────────────────────────────────────────────────────────
# 1. Download RAVDESS from Kaggle:
#    https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
#    Extract to: ECHOAI/ravdess-data/
#    Verify: ravdess-data/Actor_01/ exists with .wav files inside
#
# 2. Activate venv:
#    echomind-env\Scripts\activate
#
# 3. Install audio dependencies:
#    pip install librosa==0.10.2 soundfile==0.12.1 sounddevice==0.5.1
#
# 4. Run:
#    python train_audio.py
#
# Expected training time on RTX 4050: 20-35 minutes
# Output saved to: ./echomind-audio-output/
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}\n SECTION 1 -- IMPORTS\n{'='*60}\n")

# ── SECTION 1 -- IMPORTS ──────────────────────────────────────────────────────
import os
import re
import json
import time
import pickle
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("librosa / soundfile not installed.")
    print("Run: pip install librosa==0.10.2 soundfile==0.12.1")
    raise SystemExit(1)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from huggingface_hub import login, HfApi

warnings.filterwarnings("ignore")
print("All imports successful.")

print(f"\n{'='*60}\n SECTION 2 -- CONSTANTS AND GPU VERIFICATION\n{'='*60}\n")

# ── SECTION 2 -- CONSTANTS AND GPU VERIFICATION ───────────────────────────────

# ── PATHS ──────────────────────────────────────────────────────────────────────
# RAVDESS_PATH is resolved automatically via kagglehub below.
# If you already have the data locally, set this manually and skip the download.
RAVDESS_PATH = None          # resolved in the download block below
OUTPUT_DIR   = "./echomind-audio-output"
MODEL_DIR    = f"{OUTPUT_DIR}/model"
HUB_REPO_ID  = "echomind-audio-model"

# ── AUDIO FEATURE CONFIG ───────────────────────────────────────────────────────
SAMPLE_RATE  = 22050   # RAVDESS native sample rate
DURATION     = 3.0     # seconds -- truncate/pad all clips to this length
N_MFCC       = 40      # MFCC coefficients -- main feature
N_CHROMA     = 12      # Chroma features -- pitch class information
N_MELS       = 128     # Mel spectrogram bands
HOP_LENGTH   = 512     # samples between frames
N_FFT        = 2048    # FFT window size

# Feature vector size per frame.
# MFCC: 40, Chroma: 12, Mel: 128 -> total: 180 features per sample
FEATURE_DIM  = N_MFCC + N_CHROMA + N_MELS   # 180

# ── TRAINING CONFIG ────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 50      # CNN trains fast -- 50 epochs still ~25 mins
WEIGHT_DECAY  = 1e-4
PATIENCE      = 8       # early stopping patience
TEST_SPLIT    = 0.15
VAL_SPLIT     = 0.15
SEED          = 42

# ── REPRODUCIBILITY ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── DEVICE ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected. Training will be slower but will complete.")

# ── DATASET DOWNLOAD (auto via kagglehub) ─────────────────────────────────────
# kagglehub downloads to the local Kaggle cache on first run (~1.5GB).
# Subsequent runs reuse the cache -- no re-download needed.
# Requires: pip install kagglehub
# Kaggle credentials must be configured:
#   Method 1 (recommended): kaggle.json in C:\Users\USERNAME\.kaggle\
#              Get it from: https://www.kaggle.com/settings -> API -> Create Token
#   Method 2: set env vars KAGGLE_USERNAME and KAGGLE_KEY

try:
    import kagglehub
except ImportError:
    print("kagglehub not installed. Run: pip install kagglehub")
    raise SystemExit(1)

LOCAL_RAVDESS = "./ravdess-data"

if Path(LOCAL_RAVDESS).exists() and any(Path(LOCAL_RAVDESS).iterdir()):
    # Already extracted locally -- use it.
    RAVDESS_PATH = LOCAL_RAVDESS
    print(f"Using local RAVDESS data at: {Path(RAVDESS_PATH).resolve()}")
else:
    print("Downloading RAVDESS dataset via kagglehub...")
    print("First run: ~1.5GB download. Subsequent runs load from cache.")
    try:
        RAVDESS_PATH = kagglehub.dataset_download(
            "uwrfkaggler/ravdess-emotional-speech-audio"
        )
        print(f"Dataset downloaded to: {RAVDESS_PATH}")
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        print("Make sure kaggle.json is at C:\\Users\\USERNAME\\.kaggle\\kaggle.json")
        print("Get it from: https://www.kaggle.com/settings -> API -> Create New Token")
        raise SystemExit(1)

# Verify the path contains Actor folders.
actor_dirs = [d for d in Path(RAVDESS_PATH).rglob("Actor_*") if d.is_dir()]
if not actor_dirs:
    # kagglehub sometimes nests the data one level deeper -- detect it.
    candidates = [p for p in Path(RAVDESS_PATH).rglob("*.wav")]
    if candidates:
        RAVDESS_PATH = str(candidates[0].parent.parent)
        print(f"Adjusted RAVDESS_PATH to: {RAVDESS_PATH}")
    else:
        print(f"ERROR: No Actor_* folders or .wav files found under {RAVDESS_PATH}")
        print("Check the extracted folder structure.")
        raise SystemExit(1)

for d in [OUTPUT_DIR, MODEL_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

print(f"\nRAVDESS path: {Path(RAVDESS_PATH).resolve()}")
print(f"Output dir:   {Path(OUTPUT_DIR).resolve()}")
print(f"Feature dim:  {FEATURE_DIM}")

print(f"\n{'='*60}\n SECTION 3 -- LABEL MAPPING\n{'='*60}\n")

# ── SECTION 3 -- LABEL MAPPING ────────────────────────────────────────────────
RAVDESS_TO_ECHOMIND = {
    1: "neutral",
    2: "neutral",   # calm -> neutral (standard practice)
    3: "joy",
    4: "sadness",
    5: "anger",
    6: "fear",
    7: "disgust",
    8: "surprise",
}

ID2LABEL = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "fear",
    4: "surprise",
    5: "disgust",
    6: "neutral",
}
LABEL2ID          = {v: k for k, v in ID2LABEL.items()}
ECHOMIND_EMOTIONS = list(ID2LABEL.values())
NUM_CLASSES       = len(ID2LABEL)

print("Label mapping (RAVDESS code -> EchoMind class):")
for code, label in RAVDESS_TO_ECHOMIND.items():
    print(f"  RAVDESS {code} -> {label}")
print(f"\nTarget classes ({NUM_CLASSES}): {ECHOMIND_EMOTIONS}")

print(f"\n{'='*60}\n SECTION 4 -- FEATURE EXTRACTION\n{'='*60}\n")

# ── SECTION 4 -- AUDIO FEATURE EXTRACTION ────────────────────────────────────
def extract_features(file_path: str):
    """
    Extracts a combined feature vector from a single .wav file.

    Features:
    - MFCC (40): captures timbre and vocal tract shape.
      Most important feature for emotion -- voice quality changes with emotion.
    - Chroma (12): pitch class content -- captures melodic patterns.
    - Mel Spectrogram (128): frequency content in perceptual scale.

    Each feature is averaged over time to produce a fixed-size vector.

    Returns:
        numpy array of shape (FEATURE_DIM,) = (180,), or None on failure.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # Pad with zeros if clip is shorter than DURATION.
        target_length = int(SAMPLE_RATE * DURATION)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        else:
            y = y[:target_length]

        # MFCC.
        mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                          hop_length=HOP_LENGTH, n_fft=N_FFT)
        mfcc_mean = np.mean(mfcc, axis=1)   # (40,)

        # Chroma.
        chroma      = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA,
                                                   hop_length=HOP_LENGTH, n_fft=N_FFT)
        chroma_mean = np.mean(chroma, axis=1)  # (12,)

        # Mel spectrogram.
        mel      = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                                   hop_length=HOP_LENGTH, n_fft=N_FFT)
        mel_db   = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db, axis=1)     # (128,)

        # Concatenate: [mfcc_mean(40) + chroma_mean(12) + mel_mean(128)] = 180.
        features = np.concatenate([mfcc_mean, chroma_mean, mel_mean])
        return features.astype(np.float32)

    except Exception as e:
        print(f"  Error extracting {file_path}: {e}")
        return None


def extract_features_from_array(y: np.ndarray):
    """Same as extract_features() but accepts a numpy array instead of file path."""
    try:
        target_length = int(SAMPLE_RATE * DURATION)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        else:
            y = y[:target_length]

        mfcc   = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                       hop_length=HOP_LENGTH, n_fft=N_FFT)
        chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, n_chroma=N_CHROMA,
                                              hop_length=HOP_LENGTH, n_fft=N_FFT)
        mel    = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
                                                 hop_length=HOP_LENGTH, n_fft=N_FFT)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        features = np.concatenate([
            np.mean(mfcc,   axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel_db, axis=1),
        ])
        return features.astype(np.float32)
    except Exception:
        return None


def parse_ravdess_label(filename: str):
    """
    Parses RAVDESS filename to get EchoMind label ID.
    Format: 03-01-{emotion_code}-01-01-01-01.wav
    Emotion is the 3rd segment (index 2), 1-indexed.
    """
    parts = Path(filename).stem.split("-")
    if len(parts) < 3:
        return None
    try:
        emotion_code   = int(parts[2])
        echomind_label = RAVDESS_TO_ECHOMIND.get(emotion_code)
        if echomind_label is None:
            return None
        return LABEL2ID[echomind_label]
    except (ValueError, KeyError):
        return None


print("Feature extraction functions defined.")
print(f"Output feature vector size: {FEATURE_DIM} per sample.")

print(f"\n{'='*60}\n SECTION 5 -- LOAD AND EXTRACT ALL FEATURES\n{'='*60}\n")

# ── SECTION 5 -- LOAD AND EXTRACT ALL FEATURES ───────────────────────────────
print("Scanning RAVDESS dataset...")
wav_files = list(Path(RAVDESS_PATH).rglob("*.wav"))
print(f"Found {len(wav_files)} .wav files")

if len(wav_files) == 0:
    print("ERROR: No .wav files found.")
    print(f"Check that {RAVDESS_PATH}/Actor_01/*.wav exists.")
    raise SystemExit(1)

print(f"\nExtracting features (takes 3-8 minutes first time)...")
print(f"Feature vector size: {FEATURE_DIM} per sample")

features_list = []
labels_list   = []
skipped       = 0
t0            = time.time()

for i, wav_path in enumerate(wav_files):
    label = parse_ravdess_label(wav_path.name)
    if label is None:
        skipped += 1
        continue

    features = extract_features(str(wav_path))
    if features is None:
        skipped += 1
        continue

    features_list.append(features)
    labels_list.append(label)

    if (i + 1) % 100 == 0:
        elapsed   = time.time() - t0
        remaining = (elapsed / (i + 1)) * (len(wav_files) - i - 1)
        print(f"  {i+1}/{len(wav_files)} files  |  "
              f"{elapsed:.0f}s elapsed  |  ~{remaining:.0f}s remaining")

print(f"\nExtraction complete in {time.time() - t0:.1f}s")
print(f"  Loaded:  {len(features_list)} samples")
print(f"  Skipped: {skipped} files")

X = np.array(features_list, dtype=np.float32)
y = np.array(labels_list,   dtype=np.int64)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Label array shape:    {y.shape}")

print(f"\nClass distribution:")
dist = Counter(y.tolist())
for label_id, emotion in ID2LABEL.items():
    count = dist.get(label_id, 0)
    bar   = chr(9608) * int(count / max(len(y), 1) * 50)
    print(f"  {emotion:<12}  {count:>4}  {count/len(y)*100:>5.1f}%  {bar}")

# Normalize features -- critical for CNN, features have very different scales.
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler_path = f"{MODEL_DIR}/scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"\nScaler saved: {scaler_path}")
print("Scaler is required at inference time -- do not delete it.")

print(f"\n{'='*60}\n SECTION 6 -- DATASET AND DATALOADERS\n{'='*60}\n")

# ── SECTION 6 -- DATASET CLASS AND DATA LOADERS ───────────────────────────────
class RAVDESSDataset(Dataset):
    """PyTorch Dataset wrapping the extracted RAVDESS features."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


total      = len(X_scaled)
test_size  = int(total * TEST_SPLIT)
val_size   = int(total * VAL_SPLIT)
train_size = total - test_size - val_size

full_dataset = RAVDESSDataset(X_scaled, y)

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED),
)

# num_workers=0: Windows-safe, avoids multiprocessing spawn issues.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Dataset splits:")
print(f"  Train: {train_size:>4} samples")
print(f"  Val:   {val_size:>4} samples")
print(f"  Test:  {test_size:>4} samples")

train_labels_list = [y[i].item() for i in train_dataset.indices]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels_list,
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"\nClass weights (higher = rarer class):")
for i, w in enumerate(class_weights):
    print(f"  {ID2LABEL[i]:<12}  {w:.4f}")

print(f"\n{'='*60}\n SECTION 7 -- MODEL ARCHITECTURE\n{'='*60}\n")

# ── SECTION 7 -- MODEL ARCHITECTURE ──────────────────────────────────────────
class EmotionCNN(nn.Module):
    """
    1D Convolutional Neural Network for audio emotion classification.

    Architecture:
        Input: (batch, 1, FEATURE_DIM)

        Block 1: Conv1d(1->64,  k=3) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.3)
        Block 2: Conv1d(64->128, k=3) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.3)
        Block 3: Conv1d(128->256, k=3) -> BN -> ReLU -> AdaptiveAvgPool(1)

        Classifier: Linear(256->128) -> ReLU -> Dropout(0.4) -> Linear(128->7)

    Why 1D-CNN and not LSTM or Transformer?
    - RAVDESS has only ~1,440 samples -- transformers overfit on small datasets.
    - 1D-CNN captures local patterns in feature space efficiently.
    - Trains in 20-35 minutes vs 2+ hours for LSTM.
    - Achieves comparable accuracy on small audio datasets.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # -> (batch, 256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),               # -> (batch, 256)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)         # (batch, FEATURE_DIM) -> (batch, 1, FEATURE_DIM)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.classifier(x)  # (batch, NUM_CLASSES) raw logits


model        = EmotionCNN(input_dim=FEATURE_DIM, num_classes=NUM_CLASSES).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: EmotionCNN (1D-CNN)")
print(f"Total parameters: {total_params:,}")
print(f"Device: {next(model.parameters()).device}")
print(model)

print(f"\n{'='*60}\n SECTION 8 -- TRAINING SETUP\n{'='*60}\n")

# ── SECTION 8 -- TRAINING SETUP ───────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor,
    label_smoothing=0.1,
)
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
# Cosine annealing smoothly reduces LR to near 0 over NUM_EPOCHS.
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=1e-6,
)

print(f"Optimizer:  AdamW  (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"Loss:       CrossEntropyLoss + label_smoothing=0.1 + class weights")
print(f"Scheduler:  CosineAnnealingLR (T_max={NUM_EPOCHS})")
print(f"Epochs:     up to {NUM_EPOCHS} (early stopping patience={PATIENCE})")

print(f"\n{'='*60}\n SECTION 9 -- TRAINING LOOP\n{'='*60}\n")

# ── SECTION 9 -- TRAINING LOOP ────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / total, correct / total, f1


print(f"Starting training for up to {NUM_EPOCHS} epochs...")
print(f"Early stopping if val F1 does not improve for {PATIENCE} epochs.")
print("=" * 60)

history = {
    "train_loss": [], "train_acc": [],
    "val_loss":   [], "val_acc":   [], "val_f1": [],
    "lr": [],
}

best_val_f1    = 0.0
best_epoch     = 0
patience_count = 0
best_state     = None
t_start        = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    t_epoch = time.time()

    train_loss, train_acc          = train_epoch(model, train_loader, criterion, optimizer)
    val_loss,   val_acc,   val_f1  = eval_epoch(model, val_loader, criterion)
    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    history["lr"].append(current_lr)

    improved = val_f1 > best_val_f1
    if improved:
        best_val_f1    = val_f1
        best_epoch     = epoch
        patience_count = 0
        best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_count += 1

    epoch_time = time.time() - t_epoch
    marker     = " <-- best" if improved else ""
    print(f"Epoch {epoch:>3}/{NUM_EPOCHS}  |  "
          f"loss {train_loss:.4f}/{val_loss:.4f}  |  "
          f"acc {train_acc:.3f}/{val_acc:.3f}  |  "
          f"F1 {val_f1:.4f}  |  "
          f"{epoch_time:.1f}s{marker}")

    if patience_count >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} "
              f"(no improvement for {PATIENCE} epochs).")
        break

model.load_state_dict(best_state)
duration_min = (time.time() - t_start) / 60
print(f"\nBest model restored from epoch {best_epoch} (val F1: {best_val_f1:.4f})")
print(f"Total training time: {duration_min:.1f} minutes")

print(f"\n{'='*60}\n SECTION 10 -- TRAINING CURVES\n{'='*60}\n")

# ── SECTION 10 -- PLOT TRAINING CURVES ───────────────────────────────────────
epochs_ran  = len(history["train_loss"])
epoch_range = range(1, epochs_ran + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.patch.set_facecolor("#111110")

plots = [
    (axes[0], history["train_loss"], history["val_loss"],
     "#E8A030", "#6B9FFF", "Loss",     "Training Loss", "Validation Loss"),
    (axes[1], history["train_acc"],  history["val_acc"],
     "#E8A030", "#6B9FFF", "Accuracy", "Training Acc",  "Validation Acc"),
    (axes[2], None,                  history["val_f1"],
     None,      "#00BFA0", "F1",       None,            "Validation F1 (weighted)"),
]

for ax, train_vals, val_vals, c1, c2, ylabel, l1, l2 in plots:
    if train_vals is not None:
        ax.plot(epoch_range, train_vals, color=c1, linewidth=1.5,
                label=l1, marker="o", markersize=2)
    ax.plot(epoch_range, val_vals, color=c2, linewidth=1.5,
            label=l2, marker="o", markersize=2)
    ax.axvline(x=best_epoch, color="#2D2B27", linestyle="--", linewidth=1,
               label=f"best epoch {best_epoch}")
    ax.set_facecolor("#1C1B19")
    ax.set_xlabel("Epoch", color="#8A8278", fontsize=9)
    ax.set_ylabel(ylabel, color="#8A8278", fontsize=9)
    ax.tick_params(colors="#8A8278", labelsize=8)
    ax.legend(fontsize=8, facecolor="#1C1B19", labelcolor="#EDE5D0",
              edgecolor="#2D2B27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2D2B27")
    ax.grid(True, color="#2D2B27", linewidth=0.5, alpha=0.5)

plt.suptitle(f"EchoMind Audio -- EmotionCNN Training (best epoch: {best_epoch})",
             color="#EDE5D0", fontsize=12)
plt.tight_layout()
curves_path = f"{OUTPUT_DIR}/training_curves.png"
plt.savefig(curves_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {curves_path}")

print(f"\n{'='*60}\n SECTION 11 -- TEST SET EVALUATION\n{'='*60}\n")

# ── SECTION 11 -- TEST SET EVALUATION ────────────────────────────────────────
print("Evaluating on held-out test set...")
test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, criterion)

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch.to(device))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y_batch.numpy())

emotion_names = [ID2LABEL[i] for i in range(NUM_CLASSES)]

print(f"\n=== TEST SET RESULTS ===")
print(f"  Loss:          {test_loss:.4f}")
print(f"  Accuracy:      {test_acc:.4f}  "
      f"{'PASS' if test_acc >= 0.65 else 'BELOW TARGET (0.65)'}")
print(f"  F1 (weighted): {test_f1:.4f}  "
      f"{'PASS' if test_f1 >= 0.60 else 'BELOW TARGET (0.60)'}")

report      = classification_report(all_labels, all_preds,
                                     target_names=emotion_names, digits=4)
print(f"\n=== CLASSIFICATION REPORT ===")
print(report)

report_path = f"{OUTPUT_DIR}/classification_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("EchoMind Audio -- EmotionCNN\n")
    f.write(f"Trained: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(report)
print(f"Saved: {report_path}")

# Confusion matrix.
cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#111110")
for ax, data, title, fmt in [
    (axes[0], cm,      "Confusion Matrix -- Counts",       "d"),
    (axes[1], cm_norm, "Confusion Matrix -- Row Normalized", ".2f"),
]:
    sns.heatmap(data, annot=True, fmt=fmt,
                xticklabels=emotion_names, yticklabels=emotion_names,
                cmap="YlOrBr", ax=ax,
                linewidths=0.4, linecolor="#111110")
    ax.set_facecolor("#1C1B19")
    ax.set_title(title, color="#EDE5D0", fontsize=11, pad=10)
    ax.set_xlabel("Predicted", color="#8A8278")
    ax.set_ylabel("True",      color="#8A8278")
    ax.tick_params(colors="#8A8278")

plt.tight_layout()
cm_path = f"{OUTPUT_DIR}/confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {cm_path}")

print(f"\n{'='*60}\n SECTION 12 -- SAVE MODEL\n{'='*60}\n")

# ── SECTION 12 -- SAVE MODEL AND METADATA ────────────────────────────────────
model_weights_path = f"{MODEL_DIR}/emotion_cnn.pt"
torch.save(model.state_dict(), model_weights_path)

model_config = {
    "input_dim":   FEATURE_DIM,
    "num_classes": NUM_CLASSES,
    "id2label":    ID2LABEL,
    "label2id":    LABEL2ID,
}
config_path = f"{MODEL_DIR}/model_config.json"
with open(config_path, "w") as f:
    json.dump(model_config, f, indent=2)

feature_config = {
    "sample_rate": SAMPLE_RATE,
    "duration":    DURATION,
    "n_mfcc":      N_MFCC,
    "n_chroma":    N_CHROMA,
    "n_mels":      N_MELS,
    "hop_length":  HOP_LENGTH,
    "n_fft":       N_FFT,
    "feature_dim": FEATURE_DIM,
}
feature_config_path = f"{MODEL_DIR}/feature_config.json"
with open(feature_config_path, "w") as f:
    json.dump(feature_config, f, indent=2)

metadata = {
    "project":       "EchoMind",
    "model":         "EmotionCNN (1D-CNN)",
    "dataset":       "RAVDESS",
    "num_classes":   NUM_CLASSES,
    "id2label":      ID2LABEL,
    "feature_dim":   FEATURE_DIM,
    "test_accuracy": round(test_acc, 4),
    "test_f1":       round(test_f1, 4),
    "best_epoch":    best_epoch,
    "total_samples": len(X),
    "training_duration_min": round(duration_min, 1),
    "trained_on":    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "trained_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
}
meta_path = f"{OUTPUT_DIR}/echomind_audio_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

# Copy artifacts into model dir for Hub push.
import shutil
for artifact in ["training_curves.png", "confusion_matrix.png", "classification_report.txt"]:
    src = f"{OUTPUT_DIR}/{artifact}"
    dst = f"{MODEL_DIR}/{artifact}"
    if Path(src).exists():
        shutil.copy(src, dst)

print(f"Model saved:          {model_weights_path}")
print(f"Model config:         {config_path}")
print(f"Feature config:       {feature_config_path}")
print(f"Metadata:             {meta_path}")

print(f"\nModel directory contents:")
for fi in sorted(Path(MODEL_DIR).iterdir()):
    print(f"  {fi.name:<45} {fi.stat().st_size/1024:>8.1f} KB")

print(f"\n{'='*60}\n SECTION 13 -- INFERENCE FUNCTION AND MIC TEST\n{'='*60}\n")

# ── SECTION 13 -- INFERENCE FUNCTION AND MIC TEST ────────────────────────────
def load_model_for_inference():
    """Loads saved model, scaler, and configs for inference."""
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        inf_scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/model_config.json") as f:
        config = json.load(f)

    inf_model = EmotionCNN(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
    ).to(device)
    inf_model.load_state_dict(
        torch.load(f"{MODEL_DIR}/emotion_cnn.pt", map_location=device)
    )
    inf_model.eval()
    return inf_model, inf_scaler


inference_model, inference_scaler = load_model_for_inference()
print("Inference model loaded successfully.")


def predict_audio(audio_array, sample_rate: int = SAMPLE_RATE) -> dict:
    """
    Primary inference function for the audio pipeline.
    Same return shape as predict_text() and predict_face().

    Args:
        audio_array: numpy array of audio samples, or None
        sample_rate: sample rate of audio_array

    Returns:
        {
            "emotions":   {emotion_name: probability},  # all 7, sum to 1.0
            "dominant":   str,
            "confidence": float,
            "latency_ms": float,
            "has_audio":  bool
        }
    """
    t0 = time.time()

    neutral_result = {
        "emotions":   {e: (1.0 if e == "neutral" else 0.0) for e in ECHOMIND_EMOTIONS},
        "dominant":   "neutral",
        "confidence": 1.0,
        "latency_ms": 0.0,
        "has_audio":  False,
    }

    if audio_array is None or len(audio_array) == 0:
        return neutral_result

    try:
        # Resample if needed.
        if sample_rate != SAMPLE_RATE:
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )

        features = extract_features_from_array(audio_array)
        if features is None:
            return neutral_result

        features_scaled = inference_scaler.transform(features.reshape(1, -1))
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = inference_model(features_tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        emotions = {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}
        dominant = max(emotions, key=emotions.get)
        latency  = (time.time() - t0) * 1000

        return {
            "emotions":   emotions,
            "dominant":   dominant,
            "confidence": round(emotions[dominant], 4),
            "latency_ms": round(latency, 1),
            "has_audio":  True,
        }

    except Exception as e:
        print(f"Audio inference error: {e}")
        return neutral_result


# ── Microphone test ────────────────────────────────────────────────────────────
print("Testing inference with microphone input...")
print(f"Recording {DURATION}s of audio -- say something emotional...")

try:
    import sounddevice as sd
    recording  = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
    )
    sd.wait()
    audio_data = recording.flatten()
    print(f"Recorded {len(audio_data)} samples ({DURATION}s at {SAMPLE_RATE}Hz)")

    result = predict_audio(audio_data, sample_rate=SAMPLE_RATE)
    print(f"\n=== MIC TEST RESULT ===")
    print(f"Dominant:   {result['dominant'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Latency:    {result['latency_ms']:.0f}ms")
    print(f"\nAll emotions:")
    for emotion, prob in sorted(result["emotions"].items(),
                                key=lambda x: x[1], reverse=True):
        bar = chr(9608) * int(prob * 30)
        print(f"  {emotion:<12}  {prob:.4f}  {bar}")

except Exception as e:
    print(f"Mic test skipped: {e}")
    print("Install sounddevice: pip install sounddevice==0.5.1")
    print("Or test predict_audio() manually with a .wav file.")

print(f"\n{'='*60}\n SECTION 14 -- WRITE predict_audio.py\n{'='*60}\n")

# ── SECTION 14 -- WRITE STANDALONE predict_audio.py ──────────────────────────
predict_script = (
    '# -*- coding: utf-8 -*-\n'
    '"""\n'
    'EchoMind -- Audio inference\n'
    'Loads trained 1D-CNN emotion classifier from local model directory.\n'
    'Drop-in for Gradio app -- same return shape as predict_text() and predict_face().\n'
    '\n'
    'Requires: pip install librosa sounddevice\n'
    '"""\n'
    'import json\n'
    'import time\n'
    'import pickle\n'
    'import numpy as np\n'
    'import torch\n'
    'import torch.nn as nn\n'
    'import librosa\n'
    '\n'
    'MODEL_DIR = "./echomind-audio-output/model"\n'
    '\n'
    'with open(f"{MODEL_DIR}/model_config.json")   as f: MODEL_CONFIG   = json.load(f)\n'
    'with open(f"{MODEL_DIR}/feature_config.json") as f: FEATURE_CONFIG = json.load(f)\n'
    'with open(f"{MODEL_DIR}/scaler.pkl", "rb")    as f: SCALER         = pickle.load(f)\n'
    '\n'
    'ID2LABEL          = {int(k): v for k, v in MODEL_CONFIG["id2label"].items()}\n'
    'ECHOMIND_EMOTIONS = list(ID2LABEL.values())\n'
    'DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    '\n'
    '\n'
    'class EmotionCNN(nn.Module):\n'
    '    def __init__(self, input_dim, num_classes):\n'
    '        super().__init__()\n'
    '        self.conv_block1 = nn.Sequential(\n'
    '            nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64),\n'
    '            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),\n'
    '        )\n'
    '        self.conv_block2 = nn.Sequential(\n'
    '            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128),\n'
    '            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),\n'
    '        )\n'
    '        self.conv_block3 = nn.Sequential(\n'
    '            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256),\n'
    '            nn.ReLU(), nn.AdaptiveAvgPool1d(1),\n'
    '        )\n'
    '        self.classifier = nn.Sequential(\n'
    '            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),\n'
    '            nn.Dropout(0.4), nn.Linear(128, num_classes),\n'
    '        )\n'
    '    def forward(self, x):\n'
    '        return self.classifier(\n'
    '            self.conv_block3(self.conv_block2(self.conv_block1(x.unsqueeze(1))))\n'
    '        )\n'
    '\n'
    '\n'
    '_model = None\n'
    '\n'
    '\n'
    'def get_model():\n'
    '    global _model\n'
    '    if _model is None:\n'
    '        _model = EmotionCNN(FEATURE_CONFIG["feature_dim"],\n'
    '                            MODEL_CONFIG["num_classes"]).to(DEVICE)\n'
    '        _model.load_state_dict(torch.load(f"{MODEL_DIR}/emotion_cnn.pt",\n'
    '                                          map_location=DEVICE))\n'
    '        _model.eval()\n'
    '    return _model\n'
    '\n'
    '\n'
    'def predict_audio(audio_array, sample_rate=22050) -> dict:\n'
    '    """Same return shape as predict_text() and predict_face()."""\n'
    '    t0      = time.time()\n'
    '    neutral = {\n'
    '        "emotions":   {e: (1.0 if e == "neutral" else 0.0) for e in ECHOMIND_EMOTIONS},\n'
    '        "dominant":   "neutral",\n'
    '        "confidence": 1.0,\n'
    '        "latency_ms": 0.0,\n'
    '        "has_audio":  False,\n'
    '    }\n'
    '    if audio_array is None or len(audio_array) == 0:\n'
    '        return neutral\n'
    '    try:\n'
    '        sr = FEATURE_CONFIG["sample_rate"]\n'
    '        if sample_rate != sr:\n'
    '            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=sr)\n'
    '        target_len  = int(sr * FEATURE_CONFIG["duration"])\n'
    '        audio_array = np.pad(audio_array, (0, max(0, target_len - len(audio_array))))[:target_len]\n'
    '        mfcc   = librosa.feature.mfcc(y=audio_array, sr=sr,\n'
    '                     n_mfcc=FEATURE_CONFIG["n_mfcc"],\n'
    '                     hop_length=FEATURE_CONFIG["hop_length"],\n'
    '                     n_fft=FEATURE_CONFIG["n_fft"])\n'
    '        chroma = librosa.feature.chroma_stft(y=audio_array, sr=sr,\n'
    '                     n_chroma=FEATURE_CONFIG["n_chroma"],\n'
    '                     hop_length=FEATURE_CONFIG["hop_length"],\n'
    '                     n_fft=FEATURE_CONFIG["n_fft"])\n'
    '        mel    = librosa.power_to_db(\n'
    '                     librosa.feature.melspectrogram(y=audio_array, sr=sr,\n'
    '                         n_mels=FEATURE_CONFIG["n_mels"],\n'
    '                         hop_length=FEATURE_CONFIG["hop_length"],\n'
    '                         n_fft=FEATURE_CONFIG["n_fft"]),\n'
    '                     ref=np.max)\n'
    '        features = np.concatenate([np.mean(mfcc,1), np.mean(chroma,1), np.mean(mel,1)])\n'
    '        scaled   = SCALER.transform(features.astype(np.float32).reshape(1,-1))\n'
    '        tensor   = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)\n'
    '        with torch.no_grad():\n'
    '            probs = torch.softmax(get_model()(tensor), dim=1).cpu().numpy()[0]\n'
    '        emotions = {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(len(ID2LABEL))}\n'
    '        dominant = max(emotions, key=emotions.get)\n'
    '        return {\n'
    '            "emotions":   emotions,\n'
    '            "dominant":   dominant,\n'
    '            "confidence": round(emotions[dominant], 4),\n'
    '            "latency_ms": round((time.time() - t0) * 1000, 1),\n'
    '            "has_audio":  True,\n'
    '        }\n'
    '    except Exception as e:\n'
    '        print(f"Audio inference error: {e}")\n'
    '        return neutral\n'
    '\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    import sounddevice as sd\n'
    '    print("Recording 3s -- say something...")\n'
    '    rec = sd.rec(int(3 * 22050), samplerate=22050, channels=1, dtype=np.float32)\n'
    '    sd.wait()\n'
    '    result = predict_audio(rec.flatten())\n'
    '    print("Dominant:", result["dominant"].upper(), f"({result[\'confidence\']:.1%})")\n'
    '    print("Latency: ", result["latency_ms"], "ms")\n'
    '    for e, p in sorted(result["emotions"].items(), key=lambda x: x[1], reverse=True):\n'
    '        print(f"  {e:<12} {p:.4f}")\n'
)

predict_path = f"{OUTPUT_DIR}/predict_audio.py"
with open(predict_path, "w", encoding="utf-8") as f:
    f.write(predict_script)
print(f"Saved: {predict_path}")

print(f"\n{'='*60}\n SECTION 15 -- PUSH TO HUGGINGFACE HUB\n{'='*60}\n")

# ── SECTION 15 -- PUSH TO HUGGINGFACE HUB ────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    print("HF_TOKEN not set -- skipping Hub push.")
    print("Set with:  set HF_TOKEN=your_token  (Windows)")
    print(f"Model saved locally at: {MODEL_DIR}")
else:
    login(token=hf_token)
    api = HfApi()

    try:
        api.create_repo(repo_id=HUB_REPO_ID, exist_ok=True)
    except Exception:
        pass

    files_to_push = [
        f"{MODEL_DIR}/emotion_cnn.pt",
        f"{MODEL_DIR}/model_config.json",
        f"{MODEL_DIR}/feature_config.json",
        f"{MODEL_DIR}/scaler.pkl",
        f"{OUTPUT_DIR}/training_curves.png",
        f"{OUTPUT_DIR}/confusion_matrix.png",
        f"{OUTPUT_DIR}/classification_report.txt",
        f"{OUTPUT_DIR}/echomind_audio_metadata.json",
    ]

    for file_path in files_to_push:
        if Path(file_path).exists():
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=Path(file_path).name,
                repo_id=HUB_REPO_ID,
                repo_type="model",
            )
            print(f"Uploaded: {Path(file_path).name}")

    print(f"\nModel live at: https://huggingface.co/{HUB_REPO_ID}")

print(f"\n{'='*60}\n SECTION 16 -- FINAL SUMMARY\n{'='*60}\n")

# ── SECTION 16 -- FINAL SUMMARY ───────────────────────────────────────────────
print("\n" + "="*60)
print("  AUDIO PIPELINE COMPLETE")
print("="*60)
print(f"\n  Model:       EmotionCNN (1D-CNN)")
print(f"  Dataset:     RAVDESS ({len(X)} samples)")
print(f"  Test Acc:    {test_acc:.4f}  {'PASS' if test_acc >= 0.65 else 'BELOW TARGET'}")
print(f"  Test F1:     {test_f1:.4f}  {'PASS' if test_f1 >= 0.60 else 'BELOW TARGET'}")
print(f"  Best Epoch:  {best_epoch}")
print(f"  Duration:    {duration_min:.1f} minutes")
print(f"\n  Output files:")
for fi in sorted(Path(OUTPUT_DIR).rglob("*")):
    if fi.is_file():
        rel = str(fi.relative_to(OUTPUT_DIR))
        print(f"    {rel:<50} {fi.stat().st_size/1024:>7.1f} KB")
print(f"\n  Next: wire predict_audio() into Gradio fusion layer")
print(f"  Import:  from {OUTPUT_DIR.lstrip('./')}/predict_audio import predict_audio")
print("="*60)
