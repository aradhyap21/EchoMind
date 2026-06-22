# -*- coding: utf-8 -*-
"""
EchoMind -- NLP inference
Loads fine-tuned roberta-base from HuggingFace Hub.
Model: arp210905/echomind-emotion-model
"""
import time
from transformers import pipeline

HUB_MODEL_ID = "arp210905/echomind-emotion-model"

_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "text-classification",
            model=HUB_MODEL_ID,
            top_k=None,
            truncation=True,
            max_length=256,
            device=0,  # GPU. Use device=-1 for CPU only.
        )
    return _pipe


def predict_text(text: str) -> dict:
    """
    Same return shape as JS mock predictText() in echomind-app.html.
    Drop-in replacement -- return shape is identical.

    Returns:
        {
            "emotions":   {"joy": 0.45, "sadness": 0.12, ...},  # all 7, sum to 1.0
            "dominant":   "joy",
            "confidence": 0.45,
            "latency_ms": 23.7
        }
    """
    if not text or not text.strip():
        emotions = {
            "joy": 0, "sadness": 0, "anger": 0, "fear": 0,
            "surprise": 0, "disgust": 0, "neutral": 1.0,
        }
        return {
            "emotions":   emotions,
            "dominant":   "neutral",
            "confidence": 1.0,
            "latency_ms": 0.0,
        }

    t0  = time.time()
    raw = get_pipeline()(text.strip())[0]
    ms  = (time.time() - t0) * 1000

    total    = sum(r["score"] for r in raw)
    emotions = {r["label"]: round(r["score"] / total, 4) for r in raw}
    dominant = max(emotions, key=emotions.get)

    return {
        "emotions":   emotions,
        "dominant":   dominant,
        "confidence": round(emotions[dominant], 4),
        "latency_ms": round(ms, 1),
    }


if __name__ == "__main__":
    print("Loading model from Hub:", HUB_MODEL_ID)
    print("First call downloads weights if not cached (~500MB)...\n")

    tests = [
        "I just got the job offer!",
        "I miss my old friends so much.",
        "This is absolutely unacceptable.",
        "I went to the store and bought groceries.",
    ]

    for text in tests:
        result = predict_text(text)
        print(f"Input:     {text}")
        print(f"Dominant:  {result['dominant'].upper()} ({result['confidence']:.1%})")
        print(f"Latency:   {result['latency_ms']:.0f}ms")
        top3 = sorted(result["emotions"].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top 3:     " + "  |  ".join(f"{e} {v:.1%}" for e, v in top3))
        print()
