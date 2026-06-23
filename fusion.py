# -*- coding: utf-8 -*-
"""
EchoMind -- Fusion Layer

Imports all three modality inference functions and exposes:
  - predict_text(text)        -> dict
  - predict_face(frame)       -> dict
  - predict_audio(audio, sr)  -> dict
  - fuse(text_pred, face_pred, audio_pred, weights) -> dict

All four return the canonical EchoMind prediction shape:
    {
        "emotions":   {"joy": float, "sadness": float, ...},  # 7 keys, sum to 1.0
        "dominant":   str,
        "confidence": float,
        "latency_ms": float,
    }

Model loading is lazy -- nothing loads until the first call.
Each model falls back to a mock if it fails to load (missing files, no GPU, etc.)
so the application never crashes on startup.
"""

import time
import random
import math
import numpy as np

# ── Emotion schema (must match across all three models) ──────────────────────
EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

# ── Default fusion weights (text=0.4, face=0.35, audio=0.25) ─────────────────
DEFAULT_WEIGHTS = (0.40, 0.35, 0.25)


# ─────────────────────────────────────────────────────────────────────────────
# MOCK FALLBACKS
# Used when a real model fails to load. Identical behaviour to the JS mocks
# in echomind-app.html so the app degrades gracefully.
# ─────────────────────────────────────────────────────────────────────────────

def _dirichlet(alphas):
    """Approximate Dirichlet sample using gamma distributions."""
    gamma = []
    for a in alphas:
        s = 0.0
        for _ in range(max(1, int(a * 10))):
            s -= math.log(max(random.random(), 1e-9))
        gamma.append(s / (a * 10))
    total = sum(gamma)
    return [v / total for v in gamma]


_KEYWORD_MAP = {
    "joy":      ["happy", "great", "love", "amazing", "excited", "glad", "fun", "good"],
    "sadness":  ["sad", "miss", "cry", "alone", "lost", "hurt", "tired", "sorry"],
    "anger":    ["angry", "hate", "furious", "annoyed", "frustrated", "mad"],
    "fear":     ["scared", "afraid", "nervous", "anxious", "worried", "panic"],
    "surprise": ["wow", "unexpected", "shocked", "sudden", "wait", "really"],
    "disgust":  ["gross", "disgusting", "awful", "horrible", "nasty", "terrible"],
}

def _mock_predict_text(text: str) -> dict:
    words  = text.lower().split()
    alphas = [2, 1, 1, 1, 1, 1, 3]
    for i, emotion in enumerate(EMOTIONS):
        if emotion in _KEYWORD_MAP:
            hits = sum(1 for k in _KEYWORD_MAP[emotion] if any(k in w for w in words))
            if hits:
                alphas[i] += hits * 4
    probs    = _dirichlet(alphas)
    emotions = dict(zip(EMOTIONS, probs))
    dominant = max(emotions, key=emotions.get)
    return {"emotions": emotions, "dominant": dominant,
            "confidence": round(emotions[dominant], 4), "latency_ms": 0.0}

def _mock_predict_face(_frame) -> dict:
    t = time.time()
    base     = [abs(math.sin(t * 0.3 + i * 1.1)) for i in range(7)]
    total    = sum(base)
    probs    = [v / total for v in base]
    emotions = dict(zip(EMOTIONS, probs))
    dominant = max(emotions, key=emotions.get)
    return {"emotions": emotions, "dominant": dominant,
            "confidence": round(emotions[dominant], 4), "latency_ms": 0.0,
            "face_detected": False, "face_region": None}

def _mock_predict_audio(_audio) -> dict:
    probs    = _dirichlet([1.5, 1, 1, 1, 1, 1, 2])
    emotions = dict(zip(EMOTIONS, probs))
    dominant = max(emotions, key=emotions.get)
    return {"emotions": emotions, "dominant": dominant,
            "confidence": round(emotions[dominant], 4), "latency_ms": 0.0,
            "has_audio": False}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (lazy, with fallback)
# ─────────────────────────────────────────────────────────────────────────────

_nlp_loaded   = False
_audio_loaded = False
_face_loaded  = False
_nlp_fn       = None
_audio_fn     = None
_face_fn      = None


def _load_nlp():
    global _nlp_loaded, _nlp_fn
    if _nlp_loaded:
        return
    try:
        from predict import predict_text as _pt
        _pt("warmup")
        _nlp_fn = _pt
        print("[EchoMind] NLP model loaded (HuggingFace Hub).")
    except Exception as e:
        print(f"[EchoMind] NLP model failed to load: {e}")
        print("[EchoMind] Falling back to mock NLP.")
        _nlp_fn = _mock_predict_text
    _nlp_loaded = True


def _load_audio():
    global _audio_loaded, _audio_fn
    if _audio_loaded:
        return
    try:
        from predict_audio import predict_audio as _pa
        _pa(np.zeros(22050, dtype=np.float32), sample_rate=22050)
        _audio_fn = _pa
        print("[EchoMind] Audio model loaded (local 1D-CNN).")
    except Exception as e:
        print(f"[EchoMind] Audio model failed to load: {e}")
        print("[EchoMind] Falling back to mock audio.")
        _audio_fn = _mock_predict_audio
    _audio_loaded = True


def _load_face():
    global _face_loaded, _face_fn
    if _face_loaded:
        return
    try:
        from predict_face import predict_face as _pf
        _face_fn = _pf
        print("[EchoMind] Vision model loaded (DeepFace).")
    except Exception as e:
        print(f"[EchoMind] Vision model failed to load: {e}")
        print("[EchoMind] Falling back to mock face.")
        _face_fn = _mock_predict_face
    _face_loaded = True


def preload_all():
    """
    Call this once at app startup to load all three models eagerly.
    Prevents first-call latency spikes in the Gradio interface.
    """
    print("[EchoMind] Preloading all models...")
    _load_nlp()
    _load_audio()
    _load_face()
    print("[EchoMind] All models ready.")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INFERENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def predict_text(text: str) -> dict:
    """
    Run NLP emotion inference on a text string.
    Returns canonical EchoMind dict.
    Falls back to mock if model not loaded.
    """
    _load_nlp()
    if not text or not text.strip():
        return {
            "emotions": {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS},
            "dominant": "neutral", "confidence": 1.0, "latency_ms": 0.0,
        }
    result = _nlp_fn(text)
    # Normalise keys to EchoMind schema (model may return different key names)
    return _normalise(result)


def predict_face(frame: np.ndarray) -> dict:
    """
    Run vision emotion inference on a BGR numpy frame.
    Returns canonical EchoMind dict + face_detected + face_region.
    Falls back to mock if model not loaded or no face found.
    """
    _load_face()
    if frame is None:
        return {
            "emotions": {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS},
            "dominant": "neutral", "confidence": 1.0, "latency_ms": 0.0,
            "face_detected": False, "face_region": None,
        }
    result = _face_fn(frame)
    return _normalise(result)


def predict_audio(audio_array: np.ndarray, sample_rate: int = 22050) -> dict:
    """
    Run audio emotion inference on a float32 numpy array.
    Returns canonical EchoMind dict + has_audio.
    Falls back to mock if model not loaded or audio is empty.
    """
    _load_audio()
    if audio_array is None or len(audio_array) == 0:
        return {
            "emotions": {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS},
            "dominant": "neutral", "confidence": 1.0, "latency_ms": 0.0,
            "has_audio": False,
        }
    result = _audio_fn(audio_array, sample_rate)
    return _normalise(result)


# ─────────────────────────────────────────────────────────────────────────────
# FUSION
# ─────────────────────────────────────────────────────────────────────────────

def fuse(
    text_pred:  dict,
    face_pred:  dict,
    audio_pred: dict,
    weights: tuple = DEFAULT_WEIGHTS,
) -> dict:
    """
    Weighted late fusion of three modality predictions.

    Args:
        text_pred:   output of predict_text()
        face_pred:   output of predict_face()
        audio_pred:  output of predict_audio()
        weights:     (text_w, face_w, audio_w) — auto-normalised to sum to 1.0

    Returns:
        Canonical EchoMind dict:
        {
            "emotions":   {emotion: probability},  # all 7, sum to 1.0
            "dominant":   str,
            "confidence": float,
            "latency_ms": float,                   # sum of all three latencies
            "weights":    {"text": float, "face": float, "audio": float},
        }
    """
    t_w, f_w, a_w = weights
    total_w = t_w + f_w + a_w
    if total_w <= 0:
        total_w = 1.0
    t_w, f_w, a_w = t_w / total_w, f_w / total_w, a_w / total_w

    te = text_pred.get("emotions",  {})
    fe = face_pred.get("emotions",  {})
    ae = audio_pred.get("emotions", {})

    fused = {}
    for emotion in EMOTIONS:
        tv = te.get(emotion, 0.0)
        fv = fe.get(emotion, 0.0)
        av = ae.get(emotion, 0.0)
        fused[emotion] = round(tv * t_w + fv * f_w + av * a_w, 4)

    # Re-normalise (floating point drift)
    s = sum(fused.values())
    if s > 0:
        fused = {k: round(v / s, 4) for k, v in fused.items()}

    dominant   = max(fused, key=fused.get)
    total_lat  = (
        text_pred.get("latency_ms",  0.0) +
        face_pred.get("latency_ms",  0.0) +
        audio_pred.get("latency_ms", 0.0)
    )

    return {
        "emotions":   fused,
        "dominant":   dominant,
        "confidence": round(fused[dominant], 4),
        "latency_ms": round(total_lat, 1),
        "weights":    {"text": round(t_w, 3), "face": round(f_w, 3), "audio": round(a_w, 3)},
    }


def run_all(
    text:        str        = "",
    frame:       np.ndarray = None,
    audio_array: np.ndarray = None,
    sample_rate: int        = 22050,
    weights:     tuple      = DEFAULT_WEIGHTS,
) -> dict:
    """
    Convenience function: runs all three predictors and fuses in one call.

    Returns:
        {
            "text":   predict_text result,
            "face":   predict_face result,
            "audio":  predict_audio result,
            "fused":  fuse() result,
        }
    """
    text_pred  = predict_text(text)
    face_pred  = predict_face(frame)
    audio_pred = predict_audio(audio_array, sample_rate)
    fused      = fuse(text_pred, face_pred, audio_pred, weights)

    return {
        "text":  text_pred,
        "face":  face_pred,
        "audio": audio_pred,
        "fused": fused,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(result: dict) -> dict:
    """
    Ensure the emotions dict has exactly the 7 EchoMind keys and sums to 1.0.
    Strips any extra keys from the model output (face_region, has_audio, etc.)
    while preserving them as pass-through fields.
    """
    emotions = result.get("emotions", {})

    # Fill missing emotions with 0
    for e in EMOTIONS:
        if e not in emotions:
            emotions[e] = 0.0

    # Re-normalise
    s = sum(emotions.values())
    if s > 0:
        emotions = {k: round(v / s, 4) for k, v in emotions.items()}
    else:
        emotions = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}

    dominant = max(emotions, key=emotions.get)

    out = {
        "emotions":   emotions,
        "dominant":   dominant,
        "confidence": round(emotions[dominant], 4),
        "latency_ms": result.get("latency_ms", 0.0),
    }

    # Pass through modality-specific fields unchanged
    for extra in ("face_detected", "face_region", "has_audio"):
        if extra in result:
            out[extra] = result[extra]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EchoMind Fusion Layer -- Smoke Test")
    print("=" * 60)

    preload_all()

    print("\n--- Text only ---")
    t = predict_text("I just got the job and I am so excited about it!")
    print(f"  Dominant: {t['dominant'].upper()} ({t['confidence']:.1%})")
    print(f"  Latency:  {t['latency_ms']:.0f}ms")

    print("\n--- Audio (silent) ---")
    a = predict_audio(np.zeros(22050, dtype=np.float32))
    print(f"  Dominant: {a['dominant'].upper()} ({a['confidence']:.1%})")

    print("\n--- Face (None frame) ---")
    f = predict_face(None)
    print(f"  Dominant: {f['dominant'].upper()} ({f['confidence']:.1%})")

    print("\n--- Fusion ---")
    result = fuse(t, f, a)
    print(f"  Dominant: {result['dominant'].upper()} ({result['confidence']:.1%})")
    print(f"  Weights:  {result['weights']}")
    print(f"  Latency:  {result['latency_ms']:.0f}ms")

    print("\n--- run_all convenience ---")
    out = run_all(text="I am really frustrated with this situation.")
    print(f"  Text:  {out['text']['dominant']} ({out['text']['confidence']:.1%})")
    print(f"  Fused: {out['fused']['dominant']} ({out['fused']['confidence']:.1%})")

    print("\n--- Shape verification ---")
    for label, pred in [("text", t), ("audio", a), ("face", f), ("fused", result)]:
        keys_ok = all(e in pred["emotions"] for e in EMOTIONS)
        sum_ok  = abs(sum(pred["emotions"].values()) - 1.0) < 0.01
        print(f"  {label:<6}  7 keys: {'PASS' if keys_ok else 'FAIL'}  "
              f"sum=1.0: {'PASS' if sum_ok else 'FAIL'}")

    print("\nFusion layer ready.")
