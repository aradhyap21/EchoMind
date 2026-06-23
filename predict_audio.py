# -*- coding: utf-8 -*-
"""
EchoMind -- Audio inference
Loads trained 1D-CNN emotion classifier.
Tries HuggingFace Hub first, falls back to local ./echomind-audio-output/model/
"""
import os
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import librosa

# ── Model location ────────────────────────────────────────────────────────────
# On HuggingFace Spaces: model files are in ./audio_model/ (copied during setup)
# Locally: files are in ./echomind-audio-output/model/
_CANDIDATE_DIRS = [
    "./audio_model",
    "./echomind-audio-output/model",
]

MODEL_DIR = None
for _d in _CANDIDATE_DIRS:
    if os.path.exists(f"{_d}/emotion_cnn.pt"):
        MODEL_DIR = _d
        break

# Configs loaded lazily on first use — avoids crash if model not found at import time
MODEL_CONFIG   = None
FEATURE_CONFIG = None
SCALER         = None
ID2LABEL          = None
ECHOMIND_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_configs():
    global MODEL_CONFIG, FEATURE_CONFIG, SCALER, ID2LABEL, ECHOMIND_EMOTIONS
    if MODEL_CONFIG is not None:
        return True
    if MODEL_DIR is None:
        return False
    try:
        with open(f"{MODEL_DIR}/model_config.json")   as f: MODEL_CONFIG   = json.load(f)
        with open(f"{MODEL_DIR}/feature_config.json") as f: FEATURE_CONFIG = json.load(f)
        with open(f"{MODEL_DIR}/scaler.pkl", "rb")    as f: SCALER         = pickle.load(f)
        ID2LABEL          = {int(k): v for k, v in MODEL_CONFIG["id2label"].items()}
        ECHOMIND_EMOTIONS = list(ID2LABEL.values())
        return True
    except Exception as e:
        print(f"Audio config load error: {e}")
        return False


class EmotionCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(
            self.conv_block3(self.conv_block2(self.conv_block1(x.unsqueeze(1))))
        )


_model = None


def get_model():
    global _model
    if not _load_configs():
        return None
    if _model is None:
        _model = EmotionCNN(
            FEATURE_CONFIG["feature_dim"], MODEL_CONFIG["num_classes"]
        ).to(DEVICE)
        _model.load_state_dict(
            torch.load(f"{MODEL_DIR}/emotion_cnn.pt", map_location=DEVICE)
        )
        _model.eval()
    return _model


def predict_audio(audio_array, sample_rate: int = 22050) -> dict:
    """Same return shape as predict_text() and predict_face()."""
    t0 = time.time()
    _load_configs()
    emotions_keys = ECHOMIND_EMOTIONS
    neutral = {
        "emotions":   {e: (1.0 if e == "neutral" else 0.0) for e in emotions_keys},
        "dominant":   "neutral",
        "confidence": 1.0,
        "latency_ms": 0.0,
        "has_audio":  False,
    }
    if audio_array is None or len(audio_array) == 0:
        return neutral
    if not _load_configs() or FEATURE_CONFIG is None:
        return neutral
    try:
        sr = FEATURE_CONFIG["sample_rate"]
        if sample_rate != sr:
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=sr
            )
        target_len  = int(sr * FEATURE_CONFIG["duration"])
        audio_array = np.pad(
            audio_array, (0, max(0, target_len - len(audio_array)))
        )[:target_len]

        mfcc   = librosa.feature.mfcc(
            y=audio_array, sr=sr,
            n_mfcc=FEATURE_CONFIG["n_mfcc"],
            hop_length=FEATURE_CONFIG["hop_length"],
            n_fft=FEATURE_CONFIG["n_fft"],
        )
        chroma = librosa.feature.chroma_stft(
            y=audio_array, sr=sr,
            n_chroma=FEATURE_CONFIG["n_chroma"],
            hop_length=FEATURE_CONFIG["hop_length"],
            n_fft=FEATURE_CONFIG["n_fft"],
        )
        mel    = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=audio_array, sr=sr,
                n_mels=FEATURE_CONFIG["n_mels"],
                hop_length=FEATURE_CONFIG["hop_length"],
                n_fft=FEATURE_CONFIG["n_fft"],
            ),
            ref=np.max,
        )
        features = np.concatenate(
            [np.mean(mfcc, 1), np.mean(chroma, 1), np.mean(mel, 1)]
        )
        scaled  = SCALER.transform(features.astype(np.float32).reshape(1, -1))
        tensor  = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)
        mdl = get_model()
        if mdl is None:
            return neutral
        with torch.no_grad():
            probs = torch.softmax(mdl(tensor), dim=1).cpu().numpy()[0]

        emotions = {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(len(ID2LABEL))}
        dominant = max(emotions, key=emotions.get)
        return {
            "emotions":   emotions,
            "dominant":   dominant,
            "confidence": round(emotions[dominant], 4),
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "has_audio":  True,
        }
    except Exception as e:
        print(f"Audio inference error: {e}")
        return neutral
