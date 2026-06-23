# -*- coding: utf-8 -*-
"""
EchoMind -- FastAPI backend
Serves echomind-app.html and exposes real model prediction endpoints.

Run with:
    python server.py
    
Opens at: http://localhost:8000
"""

import io
import base64
import numpy as np
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="EchoMind API", version="0.2.0")

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
    # base64-encoded JPEG/PNG image
    image_b64: str

class FuseRequest(BaseModel):
    text: Optional[str] = ""
    audio_b64: Optional[str] = None
    sample_rate: Optional[int] = 22050
    image_b64: Optional[str] = None
    weights: Optional[list] = None


# ── Serve the HTML frontend ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_app():
    html_path = Path("echomind-app.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="echomind-app.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "models": "loaded"}


# ── Predict endpoints ─────────────────────────────────────────────────────────

@app.post("/api/predict/text")
async def predict_text_endpoint(req: TextRequest):
    try:
        result = _predict_text(req.text)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/audio")
async def predict_audio_endpoint(req: AudioRequest):
    try:
        raw = base64.b64decode(req.audio_b64)
        audio_array = np.frombuffer(raw, dtype=np.float32)
        result = _predict_audio(audio_array, sample_rate=req.sample_rate)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/face")
async def predict_face_endpoint(req: FaceRequest):
    try:
        import cv2
        raw = base64.b64decode(req.image_b64)
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/fuse")
async def fuse_endpoint(req: FuseRequest):
    try:
        # Text
        tp = _predict_text(req.text or "")

        # Audio
        if req.audio_b64:
            raw = base64.b64decode(req.audio_b64)
            audio_array = np.frombuffer(raw, dtype=np.float32)
            ap = _predict_audio(audio_array, sample_rate=req.sample_rate or 22050)
        else:
            ap = _predict_audio(None)

        # Face
        if req.image_b64:
            import cv2
            raw = base64.b64decode(req.image_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            fp = _predict_face(frame) if frame is not None else _predict_face(None)
        else:
            fp = _predict_face(None)

        weights = tuple(req.weights) if req.weights and len(req.weights) == 3 else DEFAULT_WEIGHTS
        result  = fuse(tp, fp, ap, weights=weights)

        return JSONResponse({
            "text":  tp,
            "face":  {k: v for k, v in fp.items() if k != "face_region"},
            "audio": ap,
            "fused": result,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ECHOMIND -- Backend Server")
    print("="*60)
    print("  Loading models...")
    preload_all()
    print("  Server starting at http://localhost:8000")
    print("  Open http://localhost:8000 in your browser")
    print("="*60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
