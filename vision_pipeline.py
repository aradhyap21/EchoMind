# -*- coding: utf-8 -*-
# ── SETUP ─────────────────────────────────────────────────────────────────────
# 1. Activate your venv:
#      echomind-env\Scripts\activate
#
# 2. Install dependencies:
#      pip install deepface==0.0.93 tf-keras opencv-python==4.10.0.86
#
# 3. Run in TEST mode first (single snapshot):
#      python vision_pipeline.py
#    Change MODE = "live" for real-time webcam window
#
# First run downloads ~500MB of pretrained weights automatically.
# Subsequent runs load from cache -- much faster.
#
# Controls (live mode):
#   Q -- quit
#   S -- save current frame as sample_output.png
#   P -- print current prediction to terminal
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}\n SECTION 1 -- IMPORTS AND MODE SELECTION\n{'='*60}\n")

# ── SECTION 1 -- IMPORTS AND MODE SELECTION ───────────────────────────────────
import os
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# MODE controls what the script does when run.
# "test" -> single webcam snapshot + print results + save PNG + exit
# "live" -> real-time webcam window with overlay until Q is pressed
MODE = "test"   # change to "live" for real-time mode

OUTPUT_DIR = "./echomind-vision-output"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# DeepFace import with clear error message if missing.
try:
    from deepface import DeepFace
except ImportError:
    print("DeepFace not installed.")
    print("Run: pip install deepface==0.0.93 tf-keras")
    raise SystemExit(1)

try:
    import cv2
except ImportError:
    print("OpenCV not installed.")
    print("Run: pip install opencv-python==4.10.0.86")
    raise SystemExit(1)

print(f"Mode:   {MODE.upper()}")
print(f"Output: {Path(OUTPUT_DIR).resolve()}")

print(f"\n{'='*60}\n SECTION 2 -- CONSTANTS AND LABEL MAPPING\n{'='*60}\n")

# ── SECTION 2 -- CONSTANTS AND LABEL MAPPING ──────────────────────────────────

# EchoMind 7-class emotion system.
ID2LABEL = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "fear",
    4: "surprise",
    5: "disgust",
    6: "neutral",
}
ECHOMIND_EMOTIONS = list(ID2LABEL.values())

# DeepFace label -> EchoMind label mapping.
# DeepFace returns: angry, disgust, fear, happy, sad, surprise, neutral
# EchoMind uses:    anger, disgust, fear, joy,  sadness, surprise, neutral
DEEPFACE_TO_ECHOMIND = {
    "angry":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "joy",
    "sad":      "sadness",
    "surprise": "surprise",
    "neutral":  "neutral",
}

# Emotion display colors for OpenCV overlay (BGR format).
EMOTION_COLORS_BGR = {
    "joy":      (48,  160, 232),   # amber
    "sadness":  (255, 159, 107),   # blue
    "anger":    (64,  64,  217),   # red
    "fear":     (250, 139, 167),   # purple
    "surprise": (160, 191, 0  ),   # teal
    "disgust":  (172, 239, 134),   # green
    "neutral":  (96,  101, 107),   # gray
}

# DeepFace model config.
# "opencv"      -- fastest (~30ms),  least accurate
# "ssd"         -- balanced (~80ms), good for real-time  <- default
# "retinaface"  -- most accurate (~200ms), use for snapshots
DETECTOR_BACKEND = "ssd"
EMOTION_MODEL    = "Emotion"

print(f"Detector backend: {DETECTOR_BACKEND}")
print(f"Emotion model:    {EMOTION_MODEL}")
print(f"Target emotions:  {ECHOMIND_EMOTIONS}")

print(f"\n{'='*60}\n SECTION 3 -- PREDICT_FACE\n{'='*60}\n")

# ── SECTION 3 -- CORE predict_face() FUNCTION ─────────────────────────────────
def predict_face(frame: np.ndarray, enforce_detection: bool = False) -> dict:
    """
    Primary inference function for the vision pipeline.

    Takes a BGR numpy array (OpenCV frame format).
    Returns a dict with the same shape as predict_text() from the NLP pipeline:
    {
        "emotions":      {emotion_name: probability},  # all 7 emotions, sum to 1.0
        "dominant":      str,                          # highest probability emotion
        "confidence":    float,                        # dominant emotion score 0-1
        "latency_ms":    float,                        # actual inference time in ms
        "face_detected": bool,                         # whether a face was found
        "face_region":   dict | None                   # bounding box {x,y,w,h} or None
    }

    Args:
        frame:             BGR numpy array from cv2.VideoCapture
        enforce_detection: if True, raises error when no face found.
                           if False, returns neutral distribution on no-face.
    """
    t0 = time.time()

    # Default return for no-face case.
    neutral_result = {
        "emotions":      {e: (1.0 if e == "neutral" else 0.0) for e in ECHOMIND_EMOTIONS},
        "dominant":      "neutral",
        "confidence":    1.0,
        "latency_ms":    0.0,
        "face_detected": False,
        "face_region":   None,
    }

    if frame is None or frame.size == 0:
        return neutral_result

    try:
        # Run DeepFace analysis.
        # enforce_detection=False: returns result even if face not clearly detected.
        # actions=["emotion"]: skip age/gender/race analysis (faster).
        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=enforce_detection,
            silent=True,
        )

        # DeepFace returns a list -- take the first (largest) face.
        result = results[0] if isinstance(results, list) else results

        # Extract raw DeepFace emotion scores.
        raw_emotions = result.get("emotion", {})
        face_region  = result.get("region", None)

        # Map DeepFace labels -> EchoMind labels.
        mapped = {}
        for deepface_key, echomind_key in DEEPFACE_TO_ECHOMIND.items():
            raw_val = raw_emotions.get(deepface_key, 0.0)
            # DeepFace returns percentages (0-100), convert to 0-1.
            mapped[echomind_key] = float(raw_val) / 100.0

        # Ensure all 7 EchoMind emotions are present.
        for emotion in ECHOMIND_EMOTIONS:
            if emotion not in mapped:
                mapped[emotion] = 0.0

        # Normalize to sum to exactly 1.0.
        total = sum(mapped.values())
        if total > 0:
            mapped = {k: round(v / total, 4) for k, v in mapped.items()}
        else:
            mapped = {e: (1.0 if e == "neutral" else 0.0) for e in ECHOMIND_EMOTIONS}

        dominant   = max(mapped, key=mapped.get)
        confidence = mapped[dominant]
        latency    = (time.time() - t0) * 1000

        return {
            "emotions":      mapped,
            "dominant":      dominant,
            "confidence":    round(confidence, 4),
            "latency_ms":    round(latency, 1),
            "face_detected": True,
            "face_region":   face_region,
        }

    except Exception:
        # No face detected or analysis failed -- return neutral.
        # This is expected and not an error condition.
        latency        = (time.time() - t0) * 1000
        result         = neutral_result.copy()
        result["latency_ms"] = round(latency, 1)
        return result


print("predict_face() defined.")

print(f"\n{'='*60}\n SECTION 4 -- OVERLAY DRAWING FUNCTIONS\n{'='*60}\n")

# ── SECTION 4 -- OPENCV OVERLAY DRAWING FUNCTIONS ────────────────────────────
def draw_face_box(frame: np.ndarray, region: dict,
                  dominant: str, confidence: float) -> np.ndarray:
    """
    Draws a bounding box around the detected face with emotion label.
    Corner accents added for brutalist aesthetic matching the web UI.
    """
    if region is None:
        return frame

    x = region.get("x", 0)
    y = region.get("y", 0)
    w = region.get("w", 0)
    h = region.get("h", 0)

    color = EMOTION_COLORS_BGR.get(dominant, (200, 200, 200))

    # Main bounding box -- 2px line.
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Corner accents -- brutalist style, 12px lines at each corner.
    corner_len = 12
    corner_t   = 3
    # Top-left
    cv2.line(frame, (x, y),         (x + corner_len, y),         color, corner_t)
    cv2.line(frame, (x, y),         (x, y + corner_len),         color, corner_t)
    # Top-right
    cv2.line(frame, (x+w, y),       (x+w - corner_len, y),       color, corner_t)
    cv2.line(frame, (x+w, y),       (x+w, y + corner_len),       color, corner_t)
    # Bottom-left
    cv2.line(frame, (x, y+h),       (x + corner_len, y+h),       color, corner_t)
    cv2.line(frame, (x, y+h),       (x, y+h - corner_len),       color, corner_t)
    # Bottom-right
    cv2.line(frame, (x+w, y+h),     (x+w - corner_len, y+h),     color, corner_t)
    cv2.line(frame, (x+w, y+h),     (x+w, y+h - corner_len),     color, corner_t)

    # Label.
    label_text = f"{dominant.upper()}  {confidence:.0%}"
    (text_w, text_h), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    )
    label_y = y - 10 if y - 10 > text_h else y + h + text_h + 10

    cv2.rectangle(frame,
                  (x, label_y - text_h - 6),
                  (x + text_w + 8, label_y + 4),
                  color, -1)
    cv2.putText(frame, label_text,
                (x + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (10, 10, 10), 1, cv2.LINE_AA)
    return frame


def draw_emotion_bars(frame: np.ndarray, emotions: dict) -> np.ndarray:
    """
    Draws a mini horizontal bar chart of all 7 emotions in the bottom-left corner.
    Semi-transparent dark background for readability over video.
    """
    h, w      = frame.shape[:2]
    bar_x     = 12
    bar_y0    = h - 175
    bar_h     = 8
    bar_gap   = 20
    bar_max_w = 160
    bg_pad    = 8

    # Semi-transparent dark background.
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (bar_x - bg_pad, bar_y0 - 20),
                  (bar_x + bar_max_w + 60 + bg_pad, h - 8),
                  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Sort emotions by value descending for readability.
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

    for i, (emotion, prob) in enumerate(sorted_emotions):
        y     = bar_y0 + i * bar_gap
        color = EMOTION_COLORS_BGR.get(emotion, (150, 150, 150))

        # Track (background bar).
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_w, y + bar_h), (40, 40, 40), -1)

        # Fill bar.
        fill_w = int(prob * bar_max_w)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, y), (bar_x + fill_w, y + bar_h), color, -1)

        # Emotion label.
        cv2.putText(frame, f"{emotion[:4].upper()}",
                    (bar_x + bar_max_w + 6, y + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

        # Percentage.
        cv2.putText(frame, f"{prob:.0%}",
                    (bar_x + bar_max_w + 38, y + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    return frame


def draw_fps_and_status(frame: np.ndarray, fps: float,
                        face_detected: bool, latency: float) -> np.ndarray:
    """Draws FPS, detection status, and latency in the top-right corner."""
    h, w         = frame.shape[:2]
    status_color = (160, 191, 0) if face_detected else (96, 101, 107)
    status_text  = "FACE DETECTED" if face_detected else "NO FACE"

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w - 110, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, status_text,
                (w - 130, 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, status_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{latency:.0f}ms",
                (w - 80, 64), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (100, 100, 100), 1, cv2.LINE_AA)

    # Watermark.
    cv2.putText(frame, "ECHOMIND / VISION",
                (w - 155, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (50, 50, 50), 1, cv2.LINE_AA)
    return frame


def annotate_frame(frame: np.ndarray, result: dict, fps: float = 0.0) -> np.ndarray:
    """
    Master annotation function. Mirrors frame, draws box + bars + stats.
    Returns a fully annotated frame ready to display or save.
    """
    annotated = frame.copy()

    # Mirror frame (selfie / webcam view).
    annotated = cv2.flip(annotated, 1)

    # Mirror the face region x-coordinate to match flipped frame.
    region = result.get("face_region")
    if region and result["face_detected"]:
        frame_w = frame.shape[1]
        mirrored_region = {
            "x": frame_w - region["x"] - region["w"],
            "y": region["y"],
            "w": region["w"],
            "h": region["h"],
        }
        annotated = draw_face_box(
            annotated, mirrored_region,
            result["dominant"], result["confidence"]
        )

    annotated = draw_emotion_bars(annotated, result["emotions"])
    annotated = draw_fps_and_status(
        annotated, fps, result["face_detected"], result["latency_ms"]
    )
    return annotated


print("Overlay drawing functions defined.")

print(f"\n{'='*60}\n SECTION 5 -- TEST MODE\n{'='*60}\n")

# ── SECTION 5 -- TEST MODE (SINGLE SNAPSHOT) ──────────────────────────────────
def run_test_mode():
    """
    Captures a single frame from webcam, runs predict_face(), prints results,
    saves annotated frame as sample_output.png, saves metadata JSON.
    """
    print("Opening webcam for single snapshot...")
    print("Make sure your face is visible in front of the camera.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        print("Check that no other app is using the camera.")
        return

    # Warmup -- discard first 10 frames (camera needs time to adjust exposure).
    print("Warming up camera (10 frames)...")
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("ERROR: Failed to capture frame.")
        return

    print(f"Frame captured: {frame.shape[1]}x{frame.shape[0]} px")
    print("Running DeepFace analysis (first run downloads weights ~500MB)...")
    print("This may take 30-60 seconds on first run. Subsequent runs are faster.")

    result = predict_face(frame, enforce_detection=False)

    # Print results.
    print("\n=== VISION PIPELINE RESULTS ===")
    print(f"Face detected:  {result['face_detected']}")
    print(f"Dominant:       {result['dominant'].upper()}")
    print(f"Confidence:     {result['confidence']:.1%}")
    print(f"Latency:        {result['latency_ms']:.0f}ms")
    print("\nAll emotions (sorted by score):")
    sorted_emotions = sorted(result["emotions"].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in sorted_emotions:
        bar = chr(9608) * int(prob * 30)
        print(f"  {emotion:<12}  {prob:.4f}  {bar}")

    # Return shape verification -- must match NLP predict_text() output.
    print("\n=== RETURN SHAPE VERIFICATION ===")
    required_keys = ["emotions", "dominant", "confidence", "latency_ms"]
    for key in required_keys:
        status = "PASS" if key in result else "FAIL -- missing key"
        print(f"  {key:<15}  {status}")
    emotion_keys_ok = all(e in result["emotions"] for e in ECHOMIND_EMOTIONS)
    print(f"  all 7 emotions  {'PASS' if emotion_keys_ok else 'FAIL'}")
    sum_ok = abs(sum(result["emotions"].values()) - 1.0) < 0.01
    print(f"  sum to 1.0      {'PASS' if sum_ok else 'FAIL'}")

    # Save annotated frame.
    annotated = annotate_frame(frame, result, fps=0.0)
    out_path  = f"{OUTPUT_DIR}/sample_output.png"
    cv2.imwrite(out_path, annotated)
    print(f"\nSaved annotated frame: {out_path}")

    # Save metadata.
    metadata = {
        "model":            "DeepFace",
        "detector_backend": DETECTOR_BACKEND,
        "emotion_model":    EMOTION_MODEL,
        "label_mapping":    DEEPFACE_TO_ECHOMIND,
        "sample_result":    {k: v for k, v in result.items() if k != "face_region"},
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = f"{OUTPUT_DIR}/vision_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata:        {meta_path}")
    print("\nTest mode complete. Set MODE = 'live' for real-time webcam.")


print(f"\n{'='*60}\n SECTION 6 -- LIVE MODE\n{'='*60}\n")

# ── SECTION 6 -- LIVE MODE (REAL-TIME WEBCAM) ─────────────────────────────────
def run_live_mode():
    """
    Opens a real-time webcam window with emotion overlay.
    Runs until user presses Q.

    Controls:
        Q -- quit
        S -- save current annotated frame to sample_output.png
        P -- print current prediction to terminal
    """
    print("Starting live webcam mode...")
    print("Controls: Q = quit  |  S = save frame  |  P = print prediction")
    print("Tip: first frame is slow -- DeepFace loading model weights.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    # PRED_INTERVAL controls how often DeepFace runs.
    # 0.4s = runs ~2.5 times per second.
    # Video still renders every frame for smooth display.
    # Reduce to 0.2 for faster updates, increase to 0.8 for smoother video.
    PRED_INTERVAL = 0.4

    last_pred_time = 0.0
    fps_counter    = 0
    fps_start      = time.time()
    current_fps    = 0.0
    frame_count    = 0

    # Default neutral result before first DeepFace call.
    last_result = {
        "emotions":      {e: (1.0 if e == "neutral" else 0.0) for e in ECHOMIND_EMOTIONS},
        "dominant":      "neutral",
        "confidence":    1.0,
        "latency_ms":    0.0,
        "face_detected": False,
        "face_region":   None,
    }

    print("Window opened. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame capture failed. Retrying...")
            time.sleep(0.1)
            continue

        now = time.time()

        # Run prediction every PRED_INTERVAL seconds.
        if now - last_pred_time >= PRED_INTERVAL:
            last_result    = predict_face(frame, enforce_detection=False)
            last_pred_time = now

        # Annotate and display every frame.
        annotated = annotate_frame(frame, last_result, fps=current_fps)
        cv2.imshow("EchoMind -- Vision  |  Q: quit  S: save  P: print", annotated)

        # FPS calculation.
        fps_counter += 1
        frame_count += 1
        if now - fps_start >= 1.0:
            current_fps = fps_counter / (now - fps_start)
            fps_counter = 0
            fps_start   = now

        # Keyboard controls.
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            print("Quit.")
            break
        elif key in (ord("s"), ord("S")):
            out_path = f"{OUTPUT_DIR}/sample_output.png"
            cv2.imwrite(out_path, annotated)
            print(f"Saved: {out_path}")
        elif key in (ord("p"), ord("P")):
            print(f"\nCurrent prediction:")
            print(f"  Dominant:   {last_result['dominant'].upper()}")
            print(f"  Confidence: {last_result['confidence']:.1%}")
            print(f"  Latency:    {last_result['latency_ms']:.0f}ms")
            for e, p in sorted(last_result["emotions"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {e:<12}  {p:.3f}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Total frames: {frame_count}")

print(f"\n{'='*60}\n SECTION 7 -- WRITE predict_face.py\n{'='*60}\n")

# ── SECTION 7 -- WRITE STANDALONE predict_face.py ────────────────────────────
def write_predict_file():
    """
    Writes a standalone predict_face.py to the output folder.
    Drop-in for the Gradio app -- same return shape as predict_text() from NLP.
    """
    script = (
        '# -*- coding: utf-8 -*-\n'
        '"""\n'
        'EchoMind -- Vision inference\n'
        'Uses DeepFace pretrained models for emotion detection from webcam frames.\n'
        'Drop-in for the Gradio app -- same return shape as predict_text() from NLP pipeline.\n'
        '\n'
        'Install: pip install deepface==0.0.93 tf-keras opencv-python==4.10.0.86\n'
        '"""\n'
        'import time\n'
        'import numpy as np\n'
        'from deepface import DeepFace\n'
        '\n'
        'DETECTOR_BACKEND = "ssd"\n'
        '\n'
        'DEEPFACE_TO_ECHOMIND = {\n'
        '    "angry":    "anger",\n'
        '    "disgust":  "disgust",\n'
        '    "fear":     "fear",\n'
        '    "happy":    "joy",\n'
        '    "sad":      "sadness",\n'
        '    "surprise": "surprise",\n'
        '    "neutral":  "neutral",\n'
        '}\n'
        'ECHOMIND_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]\n'
        '\n'
        '\n'
        'def predict_face(frame: np.ndarray, enforce_detection: bool = False) -> dict:\n'
        '    """\n'
        '    Args:\n'
        '        frame: BGR numpy array (OpenCV format)\n'
        '    Returns:\n'
        '        {\n'
        '            "emotions":      {emotion: probability},\n'
        '            "dominant":      str,\n'
        '            "confidence":    float,\n'
        '            "latency_ms":    float,\n'
        '            "face_detected": bool,\n'
        '            "face_region":   dict | None\n'
        '        }\n'
        '    """\n'
        '    t0 = time.time()\n'
        '    neutral = {\n'
        '        "emotions":      {e: (1.0 if e == "neutral" else 0.0) for e in ECHOMIND_EMOTIONS},\n'
        '        "dominant":      "neutral",\n'
        '        "confidence":    1.0,\n'
        '        "latency_ms":    0.0,\n'
        '        "face_detected": False,\n'
        '        "face_region":   None,\n'
        '    }\n'
        '    if frame is None or frame.size == 0:\n'
        '        return neutral\n'
        '    try:\n'
        '        results = DeepFace.analyze(\n'
        '            img_path=frame,\n'
        '            actions=["emotion"],\n'
        '            detector_backend=DETECTOR_BACKEND,\n'
        '            enforce_detection=enforce_detection,\n'
        '            silent=True,\n'
        '        )\n'
        '        result      = results[0] if isinstance(results, list) else results\n'
        '        raw_emotions = result.get("emotion", {})\n'
        '        face_region  = result.get("region", None)\n'
        '        mapped = {\n'
        '            em_key: float(raw_emotions.get(df_key, 0.0)) / 100.0\n'
        '            for df_key, em_key in DEEPFACE_TO_ECHOMIND.items()\n'
        '        }\n'
        '        for e in ECHOMIND_EMOTIONS:\n'
        '            if e not in mapped:\n'
        '                mapped[e] = 0.0\n'
        '        total  = sum(mapped.values())\n'
        '        mapped = (\n'
        '            {k: round(v / total, 4) for k, v in mapped.items()}\n'
        '            if total > 0 else\n'
        '            {e: (1.0 if e == "neutral" else 0.0) for e in ECHOMIND_EMOTIONS}\n'
        '        )\n'
        '        dominant = max(mapped, key=mapped.get)\n'
        '        return {\n'
        '            "emotions":      mapped,\n'
        '            "dominant":      dominant,\n'
        '            "confidence":    round(mapped[dominant], 4),\n'
        '            "latency_ms":    round((time.time() - t0) * 1000, 1),\n'
        '            "face_detected": True,\n'
        '            "face_region":   face_region,\n'
        '        }\n'
        '    except Exception:\n'
        '        result = neutral.copy()\n'
        '        result["latency_ms"] = round((time.time() - t0) * 1000, 1)\n'
        '        return result\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    import cv2\n'
        '    cap = cv2.VideoCapture(0)\n'
        '    for _ in range(5):\n'
        '        cap.read()\n'
        '    ret, frame = cap.read()\n'
        '    cap.release()\n'
        '    if ret:\n'
        '        result = predict_face(frame)\n'
        '        print("Dominant:", result["dominant"].upper(),\n'
        '              f"({result[\'confidence\']:.1%})")\n'
        '        print("Latency: ", result["latency_ms"], "ms")\n'
        '        for e, p in sorted(result["emotions"].items(),\n'
        '                           key=lambda x: x[1], reverse=True):\n'
        '            print(f"  {e:<12} {p:.4f}")\n'
    )

    out_path = f"{OUTPUT_DIR}/predict_face.py"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"Saved: {out_path}")


print(f"\n{'='*60}\n SECTION 8 -- MAIN ENTRY POINT\n{'='*60}\n")

# ── SECTION 8 -- MAIN ENTRY POINT ─────────────────────────────────────────────
# All execution is inside main() and guarded by __name__ == "__main__".
# This is required on Windows because DeepFace internally uses multiprocessing
# in some backends and would crash without this guard.

def main():
    print("\n" + "="*60)
    print("  ECHOMIND -- VISION PIPELINE")
    print("="*60)
    print(f"  Mode:     {MODE.upper()}")
    print(f"  Detector: {DETECTOR_BACKEND}")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    # Write standalone predict_face.py to output folder.
    write_predict_file()

    # Run selected mode.
    if MODE == "test":
        run_test_mode()
    elif MODE == "live":
        run_live_mode()
    else:
        print(f"Unknown MODE: '{MODE}'")
        print("Set MODE = 'test' or MODE = 'live' at the top of the script.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Required for Windows frozen executables.
    main()
