# -*- coding: utf-8 -*-
"""
EchoMind — FastAPI Backend
Serves echomind-app.html and exposes 4 prediction endpoints.

Run:
    uvicorn app:app --reload --port 8000

Open:
    http://localhost:8000

Install:
    pip install fastapi uvicorn python-multipart soundfile
"""

import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import base64
import numpy as np
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from fusion import (
    predict_text as _predict_text,
    predict_face as _predict_face,
    predict_audio as _predict_audio,
    fuse,
    preload_all,
    EMOTIONS,
    DEFAULT_WEIGHTS,
)

app = FastAPI(title="EchoMind API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str

class AudioRequest(BaseModel):
    # base64-encoded float32 PCM audio + sample rate
    audio_b64: str
    sample_rate: int = 22050

class FaceRequest(BaseModel):
    # base64-encoded JPEG/PNG image (data URI or raw b64)
    image_b64: str

class FuseRequest(BaseModel):
    text_emotions:  Optional[dict] = None
    face_emotions:  Optional[dict] = None
    audio_emotions: Optional[dict] = None
    weights:        Optional[list] = [0.4, 0.35, 0.25]


# ── Serve the HTML frontend ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_app():
    html_path = Path("echomind-app.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="echomind-app.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "models": "loaded"}

# Also serve on /api/health for backwards compatibility
@app.get("/api/health")
async def api_health():
    return {"status": "ok", "models": "loaded"}


# ── Predict endpoints (available at both /predict/* and /api/predict/*) ────────

@app.post("/predict/text")
@app.post("/api/predict/text")
async def predict_text_endpoint(req: TextRequest):
    try:
        result = _predict_text(req.text)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/face")
@app.post("/api/predict/face")
async def predict_face_endpoint(req: FaceRequest):
    try:
        import cv2
        # Handle both raw base64 and data URI format (e.g. "data:image/jpeg;base64,...")
        img_data = req.image_b64
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        raw = base64.b64decode(img_data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
        result = _predict_face(frame)
        # face_region may contain non-serialisable values — clean it
        if "face_region" in result and result["face_region"] is not None:
            result["face_region"] = {k: int(v) for k, v in result["face_region"].items()
                                     if isinstance(v, (int, float, np.integer))}
        return JSONResponse(result)
    except Exception as e:
        print(f"[EchoMind] Face error: {e}")
        # Return neutral fallback instead of crashing
        result = _predict_face(None)
        return JSONResponse(result)


@app.post("/predict/audio")
@app.post("/api/predict/audio")
async def predict_audio_endpoint(req: AudioRequest):
    try:
        raw = base64.b64decode(req.audio_b64)
        audio_array = np.frombuffer(raw, dtype=np.float32)
        result = _predict_audio(audio_array, sample_rate=req.sample_rate)
        return JSONResponse(result)
    except Exception as e:
        print(f"[EchoMind] Audio error: {e}")
        result = _predict_audio(None)
        return JSONResponse(result)


@app.post("/predict/audio/blob")
@app.post("/api/predict/audio/blob")
async def predict_audio_blob_endpoint(file: UploadFile = File(...)):
    """
    Accepts a raw audio file (webm, ogg, wav, mp3) as multipart upload.
    Much faster than the base64 JSON endpoint — no encoding overhead.
    """
    try:
        raw_bytes = await file.read()
        # Try soundfile first (fast, handles wav/ogg/flac)
        audio_array = None
        sample_rate = 22050
        try:
            import soundfile as sf
            import io as _io
            audio_array, sample_rate = sf.read(_io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)  # stereo → mono
        except Exception:
            pass
        # Fallback: librosa (handles webm/opus/mp3 via ffmpeg if available)
        if audio_array is None:
            try:
                import librosa
                import io as _io
                audio_array, sample_rate = librosa.load(_io.BytesIO(raw_bytes), sr=None, mono=True)
            except Exception as e:
                print(f"[EchoMind] Audio blob decode error: {e}")
        if audio_array is None or len(audio_array) == 0:
            return JSONResponse(_predict_audio(None))
        result = _predict_audio(audio_array, sample_rate=int(sample_rate))
        return JSONResponse(result)
    except Exception as e:
        print(f"[EchoMind] Audio blob error: {e}")
        return JSONResponse(_predict_audio(None))


@app.post("/predict/fuse")
@app.post("/api/predict/fuse")
async def fuse_endpoint(req: FuseRequest):
    """
    Server-side fusion endpoint.
    Accepts pre-computed emotion dicts from each modality and returns
    the weighted-average fused result.
    """
    try:
        weights = tuple(req.weights) if req.weights and len(req.weights) == 3 else DEFAULT_WEIGHTS
        total_w = sum(weights) or 1.0
        w = [x / total_w for x in weights]

        # Build emotion dicts — use neutral fallback for any missing modality
        neutral_emotions = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
        te = req.text_emotions  or neutral_emotions
        fe = req.face_emotions  or neutral_emotions
        ae = req.audio_emotions or neutral_emotions

        fused = {}
        for e in EMOTIONS:
            fused[e] = (te.get(e, 0) * w[0] +
                        fe.get(e, 0) * w[1] +
                        ae.get(e, 0) * w[2])

        # Re-normalise
        s = sum(fused.values())
        if s > 0:
            fused = {k: round(v / s, 4) for k, v in fused.items()}

        dominant = max(fused, key=fused.get)
        return JSONResponse({
            "emotions": fused,
            "dominant": dominant,
            "confidence": round(fused[dominant], 4),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("\n" + "=" * 60)
    print("  ECHOMIND — FastAPI Backend")
    print("=" * 60)
    if "--preload" in sys.argv:
        print("  Preloading models (this may take a minute)...")
        preload_all()
    else:
        print("  Models will load lazily on first request.")
        print("  Use --preload flag to load all models at startup.")
    print("  Server starting at http://localhost:8000")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

