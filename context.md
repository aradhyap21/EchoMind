# EchoMind Codebase Context

This document provides a comprehensive overview of the EchoMind multimodal emotion detection system to help AI assistants understand the codebase without re-reading all files.

## Project Overview

EchoMind is a **multimodal emotion detection system** that combines three modalities:
1. **Text** - NLP emotion classification from text input
2. **Audio** - Speech emotion recognition from audio input  
3. **Vision** - Facial emotion detection from webcam/camera frames

The system fuses predictions from all three modalities to produce a final emotion classification with confidence scores.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        echomind-app.html                         │
│                  (Frontend - Single Page App)                     │
│  - Text input panel    - Webcam video panel    - Audio panel      │
│  - Real-time emotion bars & dominant display                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                           app.py                                │
│                    (FastAPI Backend - Main)                      │
│  Endpoints: /predict/text, /predict/face, /predict/audio,        │
│              /predict/fuse, /health, /api/*                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        fusion.py                                 │
│                    (Fusion Layer - Core API)                       │
│  - predict_text()    - predict_face()    - predict_audio()        │
│  - fuse()            - run_all()          - preload_all()        │
│  - Handles lazy loading with mock fallbacks                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌───────────────┼───────────────┐
                ▼             ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   predict.py     │ │  predict_face.py │ │  predict_audio.py  │
│ (NLP - RoBERTa)  │ │ (Vision - DeepFace)│ │(Audio - 1D-CNN)  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
       │                        │                   │
       ▼                        ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ HuggingFace Hub  │ │ DeepFace pretrained│ │ Local 1D CNN model │
│ arp210905/       │ │ models           │ │ (audio_model/)    │
│ echomind-emotion-│ │ (downloads on    │ │ emotion_cnn.pt    │
│ model            │ │  first run)      │ │ model_config.json │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

## Key Files

### Backend (Python)

| File | Purpose | Key Details |
|------|---------|-------------|
| `app.py` | Main FastAPI server | Serves frontend HTML, exposes prediction endpoints at both `/predict/*` and `/api/*` paths. Supports base64 JSON and multipart blob uploads for audio. |
| `fusion.py` | Fusion layer & unified API | Core inference functions with lazy loading. All predictors return canonical schema. Includes mock fallbacks when models fail to load. |
| `predict.py` | NLP text emotion inference | Uses HuggingFace transformers pipeline with `arp210905/echomind-emotion-model` (RoBERTa-based). Returns 7-class emotion distribution. |
| `predict_face.py` | Vision face emotion inference | Uses DeepFace library with OpenCV detector backend. Maps DeepFace emotions (angry, happy, sad, etc.) to EchoMind schema via temperature sharpening (T=0.5). |
| `predict_audio.py` | Audio speech emotion inference | 1D-CNN model with MFCC+Chroma+Mel features. Lazy loads model from `./audio_model/` or `./echomind-audio-output/model/`. Runs warmup in background thread. |
| `server.py` | Alternative backend server | Older version of app.py with slightly different API structure. Includes `/api/predict/fuse` that accepts text/audio/image in one request. |
| `train_audio.py` | Audio model training script | Trains 1D-CNN on RAVDESS dataset via kagglehub. Outputs to `./echomind-audio-output/model/`. Creates `scaler.pkl`, `model_config.json`, `feature_config.json`. |
| `train_nlp.py` | NLP model training script | Fine-tunes RoBERTa on GoEmotions dataset. 5 epochs, label smoothing 0.1, class-weighted loss. Outputs to `./echomind-nlp-output/model/`. |
| `vision_pipeline.py` | Standalone vision pipeline | Test/live modes for webcam. Can run independently. Produces visible overlay output. |

### Frontend (HTML/JavaScript)

| File | Purpose |
|------|---------|
| `echomind-app.html` | Complete SPA frontend (~2000 lines). Dark brutalist aesthetic. Three-panel layout for text/face/audio inputs. Real-time camera access, waveform visualization, emotion bars. |

### Model Assets

| File/Directory | Purpose |
|----------------|---------|
| `audio_model/emotion_cnn.pt` | Trained 1D-CNN weights |
| `audio_model/model_config.json` | Label mappings (id2label), 7 classes |
| `audio_model/feature_config.json` | Feature extraction config (sample_rate=22050, duration=3.0s, MFCC=40, Chroma=12, Mel=128) |
| `audio_model/scaler.pkl` | StandardScaler for audio features (must be loaded at inference) |

### Configuration

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `Dockerfile` | HuggingFace Spaces deployment config |
| `README.md` | Brief project description |

## Prediction Schema

All predictors return the **canonical EchoMind prediction shape**:

```python
{
    "emotions": {           # All 7 emotions, probabilities sum to 1.0
        "joy": 0.45,
        "sadness": 0.12,
        "anger": 0.08,
        "fear": 0.05,
        "surprise": 0.03,
        "disgust": 0.02,
        "neutral": 0.25
    },
    "dominant": "joy",       # Highest probability emotion
    "confidence": 0.45,      # Probability of dominant emotion
    "latency_ms": 23.7,      # Inference time in milliseconds
    # Optional modality-specific fields:
    "face_detected": bool,   # (vision only)
    "face_region": dict,     # (vision only) {x, y, w, h} or None
    "has_audio": bool        # (audio only)
}
```

## Emotion Labels

The 7-class emotion system (in order, indexed 0-6):
```python
EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
```

## Default Fusion Weights

Used when fusing predictions from all three modalities:
```python
DEFAULT_WEIGHTS = (0.40, 0.35, 0.25)  # text=40%, face=35%, audio=25%
```

## API Endpoints

All endpoints accept POST requests:

| Endpoint | Request Body | Response |
|----------|--------------|----------|
| `/predict/text` | `{"text": "string"}` | Canonical prediction |
| `/predict/face` | `{"image_b64": "base64"}` | Canonical + face_region |
| `/predict/audio` | `{"audio_b64": "base64", "sample_rate": int}` | Canonical + has_audio |
| `/predict/audio/blob` | multipart form (file upload) | Canonical (faster, no encoding) |
| `/predict/fuse` | `{"text_emotions": dict, "face_emotions": dict, "audio_emotions": dict, "weights": [3 floats]}` | Fused prediction |

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (recommended: app.py)
python app.py --preload  # --preload loads models at startup (avoids first-call latency)

# Or without preloading (lazy load on first request)
python app.py

# Open in browser
http://localhost:8000
```

## Training Scripts

```bash
# Train NLP model (RoBERTa on GoEmotions)
python train_nlp.py

# Train Audio model (1D-CNN on RAVDESS)
python train_audio.py

# Test vision pipeline standalone
python vision_pipeline.py  # MODE = "live" or "test"
```

## Important Patterns

1. **Lazy Loading**: Models load on first inference call, not at import time. Prevents startup crashes.

2. **Mock Fallbacks**: If a model fails to load (missing files, no GPU, etc.), all predictors fall back to deterministic mock implementations that return reasonable neutral distributions.

3. **Graceful Degradation**: The `predict_face` endpoint returns a fallback neutral result if image decoding fails, rather than crashing.

4. **Dual Endpoint Paths**: All prediction endpoints exist at both `/predict/*` and `/api/*` for backwards compatibility.

5. **Temperature Sharpening**: Vision predictions use temperature T=0.5 to reduce DeepFace's neutral bias.

6. **Feature Concatenation**: Audio features are [MFCC_mean(40) + Chroma_mean(12) + Mel_mean(128)] = 180-dim vector.

7. **Windows Compatibility**: Training scripts use `num_workers=0` and `if __name__ == "__main__"` guards to avoid multiprocessing spawn issues.

8. **Environment Variables**: 
   - `CUDA_VISIBLE_DEVICES=-1` - Forces CPU-only inference
   - `TF_ENABLE_ONEDNN_OPTS=0` - Disables oneDNN optimizations (prevents warnings)