# -*- coding: utf-8 -*-
"""
EchoMind -- Gradio Application
Multimodal emotion intelligence: text + webcam + audio -> fused prediction.

Run:
    python app.py

Serves at: http://localhost:7860
"""

import time
import json
import numpy as np
import gradio as gr
from datetime import datetime
from collections import Counter

# ── Fusion layer (loads all three real models with mock fallback) ─────────────
from fusion import (
    predict_text,
    predict_face,
    predict_audio,
    fuse,
    preload_all,
    EMOTIONS,
    DEFAULT_WEIGHTS,
)

# ── Session state (module-level, shared across all users in single-user mode) ─
MAX_HISTORY = 120
emotion_history: list = []
session_start = time.time()

EMOTION_COLORS = {
    "joy":      "#E8A030",
    "sadness":  "#6B9FFF",
    "anger":    "#D94040",
    "fear":     "#A78BFA",
    "surprise": "#00BFA0",
    "disgust":  "#86EFAC",
    "neutral":  "#6B6560",
}

# ── Cached last predictions per modality (used for fusion) ───────────────────
_last: dict = {
    "text":  None,
    "face":  None,
    "audio": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _append_history(pred: dict, source: str):
    global emotion_history
    elapsed = time.time() - session_start
    mm = int(elapsed // 60)
    ss = int(elapsed % 60)
    emotion_history.insert(0, {
        "time":      f"{mm:02d}:{ss:02d}",
        "dominant":  pred["dominant"],
        "confidence": round(pred["confidence"], 2),
        "source":    source,
        "emotions":  pred["emotions"],
    })
    if len(emotion_history) > MAX_HISTORY:
        emotion_history.pop()


def _dominant_html(pred: dict) -> str:
    dom   = pred["dominant"]
    conf  = pred["confidence"]
    color = EMOTION_COLORS.get(dom, "#8A8278")
    return f"""
    <div style="text-align:center;padding:16px 0;">
      <div style="font-family:'Space Mono',monospace;font-size:11px;
                  color:#4A4844;letter-spacing:0.2em;margin-bottom:8px;">
        DOMINANT SIGNAL
      </div>
      <div style="font-family:'Bebas Neue',sans-serif;font-size:72px;
                  color:{color};line-height:1;">
        {dom.upper()}
      </div>
      <div style="height:4px;background:#2D2B27;margin:12px 0;border-radius:0;">
        <div style="height:100%;width:{conf*100:.1f}%;background:{color};"></div>
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:11px;color:{color};">
        {conf:.1%} confidence
      </div>
    </div>"""


def _bars_html(emotions: dict, title: str = "") -> str:
    rows = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    bars = ""
    for emotion, prob in rows:
        color = EMOTION_COLORS.get(emotion, "#8A8278")
        bars += f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          <span style="font-family:'Space Mono',monospace;font-size:11px;
                       color:#8A8278;width:70px;flex-shrink:0;">
            {emotion.upper()}
          </span>
          <div style="flex:1;height:6px;background:#2D2B27;position:relative;">
            <div style="position:absolute;left:0;top:0;height:100%;
                        width:{prob*100:.1f}%;background:{color};"></div>
          </div>
          <span style="font-family:'Space Mono',monospace;font-size:11px;
                       color:{color};width:38px;text-align:right;">
            {prob:.0%}
          </span>
        </div>"""
    header = f'<div style="font-family:Space Mono,monospace;font-size:11px;color:#4A4844;letter-spacing:0.15em;margin-bottom:10px;">{title}</div>' if title else ""
    return f'<div style="background:#111110;padding:16px;">{header}{bars}</div>'


def _modality_split_html(text_pred, face_pred, audio_pred) -> str:
    def col(label, pred):
        dom   = pred["dominant"]
        color = EMOTION_COLORS.get(dom, "#8A8278")
        top3  = sorted(pred["emotions"].items(), key=lambda x: x[1], reverse=True)[:3]
        rows  = "".join(
            f'<div style="font-family:Space Mono,monospace;font-size:9px;color:#8A8278;">'
            f'<span style="color:{EMOTION_COLORS.get(e,"#8A8278")}">{e}</span> {p:.2f}</div>'
            for e, p in top3
        )
        return f"""
        <div style="flex:1;padding:0 8px;border-right:1px solid rgba(255,255,255,0.06);">
          <div style="font-family:Space Mono,monospace;font-size:10px;color:#8A8278;
                      margin-bottom:4px;">{label}</div>
          <div style="font-family:Bebas Neue,sans-serif;font-size:24px;
                      color:{color};line-height:1;margin-bottom:6px;">{dom.upper()}</div>
          {rows}
        </div>"""

    return f"""
    <div style="background:#111110;padding:12px;">
      <div style="font-family:Space Mono,monospace;font-size:11px;color:#4A4844;
                  letter-spacing:0.15em;margin-bottom:10px;">MODALITY SPLIT</div>
      <div style="display:flex;gap:0;">
        {col("TEXT", text_pred)}
        {col("FACE", face_pred)}
        {col("AUDIO", audio_pred)}
      </div>
    </div>"""


def _session_log_html() -> str:
    if not emotion_history:
        return '<div style="font-family:Space Mono,monospace;font-size:11px;color:#4A4844;padding:12px;">No analyses yet.</div>'
    rows = ""
    for entry in emotion_history[:20]:
        color = EMOTION_COLORS.get(entry["dominant"], "#8A8278")
        rows += f"""
        <div style="display:grid;grid-template-columns:48px 1fr 48px 56px;gap:4px;
                    padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06);
                    font-family:Space Mono,monospace;font-size:10px;align-items:center;">
          <span style="color:#4A4844;">{entry["time"]}</span>
          <span style="color:{color};font-weight:700;">{entry["dominant"].upper()}</span>
          <span style="color:#8A8278;text-align:right;">{entry["confidence"]}</span>
          <span style="color:#4A4844;text-align:right;font-size:9px;">{entry["source"]}</span>
        </div>"""
    return f'<div style="background:#111110;padding:12px;">{rows}</div>'


def _stats_html() -> str:
    if not emotion_history:
        return '<div style="font-family:Space Mono,monospace;font-size:11px;color:#4A4844;padding:12px;">No data yet.</div>'
    total    = len(emotion_history)
    dom_cnt  = Counter(e["dominant"] for e in emotion_history)
    dom_top  = dom_cnt.most_common(1)[0][0]
    avg_conf = sum(e["confidence"] for e in emotion_history) / total
    elapsed  = time.time() - session_start
    dur      = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"

    def card(label, value, color="#E8A030"):
        return f"""
        <div style="background:#1C1B19;border:1px solid rgba(255,255,255,0.06);
                    padding:16px;text-align:center;">
          <div style="font-family:Space Mono,monospace;font-size:10px;color:#4A4844;
                      margin-bottom:6px;">{label}</div>
          <div style="font-family:Bebas Neue,sans-serif;font-size:40px;
                      color:{color};line-height:1;">{value}</div>
        </div>"""

    dom_color = EMOTION_COLORS.get(dom_top, "#8A8278")
    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;padding:12px;background:#111110;">
      {card("TOTAL ANALYSES", str(total))}
      {card("DOMINANT", dom_top.upper(), dom_color)}
      {card("AVG CONFIDENCE", f"{avg_conf:.0%}")}
      {card("SESSION", dur)}
    </div>"""


def _neutral_pred() -> dict:
    return {
        "emotions":   {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS},
        "dominant":   "neutral",
        "confidence": 1.0,
        "latency_ms": 0.0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# TAB HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def handle_text(text: str, w_text: float, w_face: float, w_audio: float):
    """Analyze text input, fuse with last face/audio, update all outputs."""
    if not text or not text.strip():
        return [gr.update()] * 6

    text_pred  = predict_text(text)
    _last["text"] = text_pred

    face_pred  = _last["face"]  or _neutral_pred()
    audio_pred = _last["audio"] or _neutral_pred()
    fused      = fuse(text_pred, face_pred, audio_pred, (w_text, w_face, w_audio))

    _append_history(fused, "TEXT")

    return (
        _dominant_html(fused),
        _bars_html(fused["emotions"], "SIGNAL BREAKDOWN"),
        _modality_split_html(text_pred, face_pred, audio_pred),
        _session_log_html(),
        f"TEXT {text_pred['latency_ms']:.0f}ms  /  FACE {face_pred['latency_ms']:.0f}ms  /  AUDIO {audio_pred['latency_ms']:.0f}ms",
        _stats_html(),
    )


def handle_webcam(frame: np.ndarray, w_text: float, w_face: float, w_audio: float):
    """Process a webcam frame (called by gr.Image stream every ~1.2s)."""
    if frame is None:
        return [gr.update()] * 6

    face_pred  = predict_face(frame)
    _last["face"] = face_pred

    text_pred  = _last["text"]  or _neutral_pred()
    audio_pred = _last["audio"] or _neutral_pred()
    fused      = fuse(text_pred, face_pred, audio_pred, (w_text, w_face, w_audio))

    _append_history(fused, "FACE")

    return (
        _dominant_html(fused),
        _bars_html(fused["emotions"], "SIGNAL BREAKDOWN"),
        _modality_split_html(text_pred, face_pred, audio_pred),
        _session_log_html(),
        f"TEXT {text_pred['latency_ms']:.0f}ms  /  FACE {face_pred['latency_ms']:.0f}ms  /  AUDIO {audio_pred['latency_ms']:.0f}ms",
        _stats_html(),
    )


def handle_audio(audio_tuple, w_text: float, w_face: float, w_audio: float):
    """Process audio upload/record tuple (sample_rate, numpy_array)."""
    if audio_tuple is None:
        return [gr.update()] * 6

    sr, arr = audio_tuple
    # Gradio returns int16 — convert to float32
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32) / 32768.0
    # Stereo -> mono
    if arr.ndim == 2:
        arr = arr.mean(axis=1)

    audio_pred = predict_audio(arr, sample_rate=sr)
    _last["audio"] = audio_pred

    text_pred = _last["text"] or _neutral_pred()
    face_pred = _last["face"] or _neutral_pred()
    fused     = fuse(text_pred, face_pred, audio_pred, (w_text, w_face, w_audio))

    _append_history(fused, "AUDIO")

    return (
        _dominant_html(fused),
        _bars_html(fused["emotions"], "SIGNAL BREAKDOWN"),
        _modality_split_html(text_pred, face_pred, audio_pred),
        _session_log_html(),
        f"TEXT {text_pred['latency_ms']:.0f}ms  /  FACE {face_pred['latency_ms']:.0f}ms  /  AUDIO {audio_pred['latency_ms']:.0f}ms",
        _stats_html(),
    )


def handle_fuse_button(w_text: float, w_face: float, w_audio: float):
    """Re-fuse with current weight values and last modality predictions."""
    text_pred  = _last["text"]  or _neutral_pred()
    face_pred  = _last["face"]  or _neutral_pred()
    audio_pred = _last["audio"] or _neutral_pred()
    fused      = fuse(text_pred, face_pred, audio_pred, (w_text, w_face, w_audio))
    _append_history(fused, "FUSED")

    return (
        _dominant_html(fused),
        _bars_html(fused["emotions"], "SIGNAL BREAKDOWN"),
        _modality_split_html(text_pred, face_pred, audio_pred),
        _session_log_html(),
        f"TEXT {text_pred['latency_ms']:.0f}ms  /  FACE {face_pred['latency_ms']:.0f}ms  /  AUDIO {audio_pred['latency_ms']:.0f}ms",
        _stats_html(),
    )


def handle_reset():
    global emotion_history, session_start, _last
    emotion_history = []
    session_start   = time.time()
    _last           = {"text": None, "face": None, "audio": None}
    neutral = _neutral_pred()
    return (
        _dominant_html(neutral),
        _bars_html(neutral["emotions"], "SIGNAL BREAKDOWN"),
        _modality_split_html(neutral, neutral, neutral),
        _session_log_html(),
        "",
        _stats_html(),
    )


def handle_stats_tick():
    return _stats_html(), _session_log_html()


# Deep Text Analysis tab
def handle_deep_text(text: str):
    if not text or not text.strip():
        return gr.update(), gr.update()
    pred = predict_text(text)
    _last["text"] = pred
    return _dominant_html(pred), _bars_html(pred["emotions"], "TEXT BREAKDOWN")


# Audio Analysis tab
def handle_audio_tab(audio_tuple):
    if audio_tuple is None:
        return gr.update(), gr.update()
    sr, arr = audio_tuple
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32) / 32768.0
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    pred = predict_audio(arr, sample_rate=sr)
    _last["audio"] = pred
    return _dominant_html(pred), _bars_html(pred["emotions"], "AUDIO BREAKDOWN")
