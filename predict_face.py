# -*- coding: utf-8 -*-
"""
EchoMind -- Vision inference
Uses DeepFace pretrained models for emotion detection from webcam frames.
Drop-in for the Gradio app -- same return shape as predict_text() from NLP pipeline.

Install: pip install deepface==0.0.93 tf-keras opencv-python==4.10.0.86
"""
import time
import numpy as np
from deepface import DeepFace

DETECTOR_BACKEND = "opencv"

DEEPFACE_TO_ECHOMIND = {
    "angry":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "joy",
    "sad":      "sadness",
    "surprise": "surprise",
    "neutral":  "neutral",
}
ECHOMIND_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


def predict_face(frame: np.ndarray, enforce_detection: bool = False) -> dict:
    """
    Args:
        frame: BGR numpy array (OpenCV format)
    Returns:
        {
            "emotions":      {emotion: probability},
            "dominant":      str,
            "confidence":    float,
            "latency_ms":    float,
            "face_detected": bool,
            "face_region":   dict | None
        }
    """
    t0 = time.time()
    neutral = {
        "emotions":      {e: (1.0 if e == "neutral" else 0.0) for e in ECHOMIND_EMOTIONS},
        "dominant":      "neutral",
        "confidence":    1.0,
        "latency_ms":    0.0,
        "face_detected": False,
        "face_region":   None,
    }
    if frame is None or frame.size == 0:
        return neutral
    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=enforce_detection,
            silent=True,
        )
        result      = results[0] if isinstance(results, list) else results
        raw_emotions = result.get("emotion", {})
        face_region  = result.get("region", None)

        # Temperature sharpening: amplifies confident non-neutral signals
        # T < 1.0 sharpens the distribution (reduces neutral bias from DeepFace)
        TEMPERATURE = 0.5
        import math
        raw_vals = [raw_emotions.get(k, 0.0) / 100.0 for k in DEEPFACE_TO_ECHOMIND]
        # softmax(x/T) with numerical stability
        scaled  = [v / TEMPERATURE for v in raw_vals]
        max_s   = max(scaled)
        exps    = [math.exp(v - max_s) for v in scaled]
        sum_e   = sum(exps)
        sharpened = [e / sum_e for e in exps]

        mapped = {
            em_key: float(sharpened[i])
            for i, (df_key, em_key) in enumerate(DEEPFACE_TO_ECHOMIND.items())
        }
        for e in ECHOMIND_EMOTIONS:
            if e not in mapped:
                mapped[e] = 0.0
        total  = sum(mapped.values())
        mapped = (
            {k: round(v / total, 4) for k, v in mapped.items()}
            if total > 0 else
            {e: (1.0 if e == "neutral" else 0.0) for e in ECHOMIND_EMOTIONS}
        )
        dominant = max(mapped, key=mapped.get)
        return {
            "emotions":      mapped,
            "dominant":      dominant,
            "confidence":    round(mapped[dominant], 4),
            "latency_ms":    round((time.time() - t0) * 1000, 1),
            "face_detected": True,
            "face_region":   face_region,
        }
    except Exception:
        result = neutral.copy()
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return result


if __name__ == "__main__":
    import cv2
    cap = cv2.VideoCapture(0)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if ret:
        result = predict_face(frame)
        print("Dominant:", result["dominant"].upper(),
              f"({result['confidence']:.1%})")
        print("Latency: ", result["latency_ms"], "ms")
        for e, p in sorted(result["emotions"].items(),
                           key=lambda x: x[1], reverse=True):
            print(f"  {e:<12} {p:.4f}")
