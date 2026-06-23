# -*- coding: utf-8 -*-
"""
EchoMind -- Gradio Application
Run with: python app.py
Opens at: http://localhost:7860
"""

import time
import numpy as np
import gradio as gr

from fusion import (
    predict_text,
    predict_face,
    predict_audio,
    fuse,
    preload_all,
    EMOTIONS,
    DEFAULT_WEIGHTS,
)

# ── Session state ─────────────────────────────────────────────────────────────
emotion_history: list[dict] = []
MAX_HISTORY = 120
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

EMOTION_EMOJIS = {
    "joy": "😊", "sadness": "😢", "anger": "😠",
    "fear": "😨", "surprise": "😲", "disgust": "🤢", "neutral": "😐",
}


def _append_history(fused: dict, source: str):
    global emotion_history
    entry = {
        "timestamp": time.time(),
        "emotions":  fused["emotions"],
        "dominant":  fused["dominant"],
        "source":    source,
    }
    emotion_history.insert(0, entry)
    if len(emotion_history) > MAX_HISTORY:
        emotion_history = emotion_history[:MAX_HISTORY]


def _dominant_html(dominant: str, confidence: float) -> str:
    color = EMOTION_COLORS.get(dominant, "#9b9bb4")
    emoji = EMOTION_EMOJIS.get(dominant, "")
    return f"""
<div style="text-align:center;padding:16px;background:#1a1d2e;border:1px solid {color}33">
  <div style="font-size:40px">{emoji}</div>
  <div style="font-family:'Space Grotesk',sans-serif;font-size:28px;
              font-weight:700;color:{color};margin:4px 0">{dominant.upper()}</div>
  <div style="font-family:'Space Mono',monospace;font-size:14px;color:#8A8278">
    {confidence:.1%} confidence
  </div>
</div>"""


def _bars_html(emotions: dict) -> str:
    rows = ""
    for emotion in sorted(emotions, key=emotions.get, reverse=True):
        p     = emotions[emotion]
        color = EMOTION_COLORS.get(emotion, "#9b9bb4")
        rows += f"""
<div style="display:flex;align-items:center;gap:8px;margin:4px 0">
  <span style="font-family:'Space Mono',monospace;font-size:11px;
               color:#8A8278;width:68px;flex-shrink:0">{emotion.upper()}</span>
  <div style="flex:1;height:6px;background:#2D2B27">
    <div style="width:{p*100:.1f}%;height:100%;background:{color}"></div>
  </div>
  <span style="font-family:'Space Mono',monospace;font-size:11px;
               color:#8A8278;width:36px;text-align:right">{p:.0%}</span>
</div>"""
    return f'<div style="padding:8px 0">{rows}</div>'


def _session_stats_html() -> str:
    if not emotion_history:
        return "<p style='color:#4A4844;font-family:Space Mono,monospace;font-size:12px'>No analyses yet.</p>"

    count  = len(emotion_history)
    from collections import Counter
    dom_counts = Counter(e["dominant"] for e in emotion_history)
    top_dom    = dom_counts.most_common(1)[0][0]
    avg_conf   = sum(
        e["emotions"].get(e["dominant"], 0) for e in emotion_history
    ) / count
    elapsed    = int(time.time() - session_start)
    mm, ss     = divmod(elapsed, 60)

    rows = ""
    for entry in emotion_history[:20]:
        t   = int(entry["timestamp"] - session_start)
        em  = entry["dominant"]
        col = EMOTION_COLORS.get(em, "#9b9bb4")
        conf = entry["emotions"].get(em, 0)
        rows += f"""
<div style="display:grid;grid-template-columns:48px 1fr 52px 60px;
            gap:4px;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);
            font-family:Space Mono,monospace;font-size:10px">
  <span style="color:#4A4844">{mm:02d}:{t%60:02d}</span>
  <span style="color:{col};font-weight:700">{em.upper()}</span>
  <span style="color:#8A8278;text-align:right">{conf:.2f}</span>
  <span style="color:#4A4844;text-align:right">{entry['source']}</span>
</div>"""

    return f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:16px">
  <div style="background:#1a1d2e;padding:12px;text-align:center">
    <div style="font-family:Bebas Neue,sans-serif;font-size:36px;color:#E8A030">{count}</div>
    <div style="font-family:Space Mono,monospace;font-size:10px;color:#4A4844">ANALYSES</div>
  </div>
  <div style="background:#1a1d2e;padding:12px;text-align:center">
    <div style="font-family:Bebas Neue,sans-serif;font-size:36px;
                color:{EMOTION_COLORS.get(top_dom,'#9b9bb4')}">{top_dom.upper()}</div>
    <div style="font-family:Space Mono,monospace;font-size:10px;color:#4A4844">DOMINANT</div>
  </div>
  <div style="background:#1a1d2e;padding:12px;text-align:center">
    <div style="font-family:Bebas Neue,sans-serif;font-size:36px;color:#00BFA0">{avg_conf:.0%}</div>
    <div style="font-family:Space Mono,monospace;font-size:10px;color:#4A4844">AVG CONF</div>
  </div>
  <div style="background:#1a1d2e;padding:12px;text-align:center">
    <div style="font-family:Bebas Neue,sans-serif;font-size:36px;color:#EDE5D0">{mm:02d}:{ss:02d}</div>
    <div style="font-family:Space Mono,monospace;font-size:10px;color:#4A4844">DURATION</div>
  </div>
</div>
<div style="font-family:Space Mono,monospace;font-size:11px;color:#4A4844;
            margin-bottom:4px">SESSION LOG — LAST 20</div>
<div style="max-height:260px;overflow-y:auto">{rows}</div>"""


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

body, .gradio-container {
    background: #080706 !important;
    color: #EDE5D0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

/* Tab bar */
.tab-nav button {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    color: #4A4844 !important;
    background: #111110 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
.tab-nav button.selected {
    color: #E8A030 !important;
    border-bottom: 2px solid #E8A030 !important;
    background: #1C1B19 !important;
}

/* Inputs */
textarea, input[type=text] {
    background: #1C1B19 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    color: #EDE5D0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important;
}
textarea:focus, input[type=text]:focus {
    border-color: rgba(232,160,48,0.4) !important;
}

/* Buttons */
button.primary {
    background: #E8A030 !important;
    color: #080706 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    border: none !important;
}
button.primary:hover { background: #C8893C !important; }

button.secondary {
    background: transparent !important;
    border: 1px solid rgba(0,191,160,0.3) !important;
    color: #00BFA0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}

/* Sliders */
input[type=range] { accent-color: #E8A030; }

/* Plots */
.plot-container { background: transparent !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #111110; }
::-webkit-scrollbar-thumb { background: #E8A030; }

/* Labels */
.label-wrap span, label span {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    color: #4A4844 !important;
    letter-spacing: 0.1em !important;
}

/* Panel backgrounds */
.panel { background: #111110 !important; }
.block { background: #111110 !important; border: 1px solid rgba(255,255,255,0.06) !important; }

/* DEMO banner */
.demo-banner {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #00BFA0;
    letter-spacing: 0.15em;
    text-align: center;
    padding: 6px;
    background: rgba(0,191,160,0.08);
    border-bottom: 1px solid rgba(0,191,160,0.2);
}
"""

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    css=CSS,
    title="EchoMind — Multimodal Emotion Intelligence",
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Space Grotesk"),
    ),
) as demo:

    # State that persists across callbacks within a session
    txt_pred_state   = gr.State(None)
    face_pred_state  = gr.State(None)
    audio_pred_state = gr.State(None)
    weight_state     = gr.State(list(DEFAULT_WEIGHTS))

    gr.HTML("""
    <div class="demo-banner">
      ECHOMIND &nbsp;/&nbsp; MULTIMODAL EMOTION INTELLIGENCE &nbsp;/&nbsp; DEMO MODE ACTIVE
    </div>
    <div style="padding:20px 0 8px;text-align:center">
      <span style="font-family:'Bebas Neue',sans-serif;font-size:48px;
                   color:#E8A030;letter-spacing:0.05em">ECHO</span><span
            style="font-family:'Bebas Neue',sans-serif;font-size:48px;
                   color:#EDE5D0;letter-spacing:0.05em">MIND</span>
    </div>
    """)

    # ── TAB 1 — LIVE ANALYSIS ─────────────────────────────────────────────────
    with gr.Tab("LIVE ANALYSIS"):
        with gr.Row():
            # Left column — inputs
            with gr.Column(scale=4):
                gr.Markdown("##### SIGNAL 01 / TEXT", elem_classes="label-wrap")
                text_input = gr.Textbox(
                    placeholder="type something. anything.",
                    lines=4,
                    label="",
                    show_label=False,
                )
                with gr.Row():
                    text_w_slider = gr.Slider(0.1, 0.9, value=0.40, step=0.05,
                                              label="TEXT WEIGHT")
                analyze_text_btn = gr.Button("ANALYZE TEXT", variant="primary")

                gr.Markdown("##### SIGNAL 02 / WEBCAM", elem_classes="label-wrap")
                webcam_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="",
                    show_label=False,
                    type="numpy",
                    mirror_webcam=True,
                )
                with gr.Row():
                    face_w_slider = gr.Slider(0.1, 0.9, value=0.35, step=0.05,
                                              label="FACE WEIGHT")
                analyze_face_btn = gr.Button("ANALYZE FRAME", variant="secondary")

                gr.Markdown("##### SIGNAL 03 / AUDIO", elem_classes="label-wrap")
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="",
                    show_label=False,
                )
                with gr.Row():
                    audio_w_slider = gr.Slider(0.1, 0.9, value=0.25, step=0.05,
                                               label="AUDIO WEIGHT")
                analyze_audio_btn = gr.Button("ANALYZE AUDIO", variant="secondary")

            # Center column — primary output
            with gr.Column(scale=4):
                gr.Markdown(
                    "<span style='font-family:Space Mono,monospace;font-size:11px;"
                    "color:#4A4844;letter-spacing:0.2em'>DOMINANT SIGNAL</span>"
                )
                dominant_display = gr.HTML(_dominant_html("neutral", 1.0))
                confidence_bars  = gr.HTML(_bars_html(
                    {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
                ))

                fuse_btn = gr.Button("FUSE + ANALYZE", variant="secondary")
                latency_info = gr.Markdown(
                    "<span style='font-family:Space Mono,monospace;font-size:10px;"
                    "color:#4A4844'>TEXT --ms  /  FACE --ms  /  AUDIO --ms</span>"
                )

                gr.Markdown(
                    "<span style='font-family:Space Mono,monospace;font-size:11px;"
                    "color:#4A4844;letter-spacing:0.2em'>FUSION WEIGHTS</span>"
                )
                weight_display = gr.HTML()

            # Right column — secondary output
            with gr.Column(scale=3):
                gr.Markdown(
                    "<span style='font-family:Space Mono,monospace;font-size:11px;"
                    "color:#4A4844;letter-spacing:0.15em'>MODALITY SPLIT</span>"
                )
                modality_split = gr.HTML()
                gr.Markdown(
                    "<span style='font-family:Space Mono,monospace;font-size:11px;"
                    "color:#4A4844;letter-spacing:0.15em'>SESSION LOG</span>"
                )
                session_log = gr.HTML(_session_stats_html())

    # ── TAB 2 — DEEP TEXT ANALYSIS ────────────────────────────────────────────
    with gr.Tab("DEEP TEXT"):
        with gr.Row():
            with gr.Column(scale=5):
                text_input_2 = gr.Textbox(
                    placeholder="enter text for detailed analysis.",
                    lines=6,
                    label="TEXT INPUT",
                )
                analyze_text_btn_2 = gr.Button("ANALYZE", variant="primary")
            with gr.Column(scale=5):
                dominant_display_2 = gr.HTML(_dominant_html("neutral", 1.0))
                confidence_bars_2  = gr.HTML(_bars_html(
                    {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
                ))

    # ── TAB 3 — AUDIO ANALYSIS ────────────────────────────────────────────────
    with gr.Tab("AUDIO ANALYSIS"):
        with gr.Row():
            with gr.Column(scale=5):
                audio_input_3 = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="AUDIO INPUT",
                )
                analyze_audio_btn_3 = gr.Button("ANALYZE", variant="primary")
            with gr.Column(scale=5):
                dominant_display_3 = gr.HTML(_dominant_html("neutral", 1.0))
                confidence_bars_3  = gr.HTML(_bars_html(
                    {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
                ))

    # ── TAB 4 — SESSION STATS ─────────────────────────────────────────────────
    with gr.Tab("SESSION STATS"):
        refresh_btn   = gr.Button("REFRESH STATS", variant="secondary")
        stats_display = gr.HTML(_session_stats_html())

    # ── TAB 5 — ABOUT ─────────────────────────────────────────────────────────
    with gr.Tab("ABOUT"):
        gr.HTML("""
        <div style="max-width:800px;margin:0 auto;padding:32px 0;
                    font-family:'Space Grotesk',sans-serif">
          <h2 style="font-family:'Bebas Neue',sans-serif;font-size:48px;
                     color:#E8A030;letter-spacing:0.05em">ECHOMIND</h2>
          <p style="color:#8A8278;font-size:16px;line-height:1.7;margin-bottom:24px">
            Multimodal emotion intelligence. Reads emotion from text, webcam, and audio
            simultaneously. The useful signal is not when all three agree — it is when
            they conflict.
          </p>
          <div style="border:1px solid rgba(255,255,255,0.06);padding:24px;
                      margin-bottom:24px">
            <div style="font-family:'Space Mono',monospace;font-size:11px;
                        color:#00BFA0;letter-spacing:0.2em;margin-bottom:12px">
              SYSTEM STATUS
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:12px;
                        color:#8A8278;line-height:2">
              NLP MODEL &nbsp;&nbsp;&nbsp;&nbsp; arp210905/echomind-emotion-model (RoBERTa-base)<br>
              AUDIO MODEL &nbsp;&nbsp; EmotionCNN 1D-CNN on RAVDESS<br>
              VISION MODEL &nbsp; DeepFace pretrained<br>
              FUSION &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Weighted late fusion (text=0.4, face=0.35, audio=0.25)<br>
              BUILD &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.2.0 — Phase 02
            </div>
          </div>
          <p style="color:#4A4844;font-family:'Space Mono',monospace;font-size:11px">
            NOT PRODUCTION READY &nbsp;/&nbsp; MIT LICENSE
          </p>
        </div>
        """)


# ── Callback helpers ──────────────────────────────────────────────────────────

def _weights_html(tw, fw, aw):
    total = tw + fw + aw or 1.0
    tw_n, fw_n, aw_n = tw/total, fw/total, aw/total
    rows = ""
    for label, val, color in [
        ("TEXT",  tw_n, "#E8A030"),
        ("FACE",  fw_n, "#00BFA0"),
        ("AUDIO", aw_n, "#8A8278"),
    ]:
        rows += f"""
<div style="display:flex;align-items:center;gap:8px;margin:4px 0">
  <span style="font-family:Space Mono,monospace;font-size:11px;
               color:#8A8278;width:48px">{label}</span>
  <div style="flex:1;height:4px;background:#2D2B27">
    <div style="width:{val*100:.1f}%;height:100%;background:{color}"></div>
  </div>
  <span style="font-family:Space Mono,monospace;font-size:11px;
               color:{color};width:36px;text-align:right">{val:.0%}</span>
</div>"""
    norm_warn = "" if abs(tw+fw+aw - 1.0) < 0.05 else (
        "<div style='font-family:Space Mono,monospace;font-size:10px;"
        "color:#E8A030;margin-top:4px'>AUTO-NORMALIZING</div>"
    )
    return f"<div>{rows}{norm_warn}</div>"


def _modality_html(tp, fp, ap):
    html = '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">'
    for label, pred in [("TEXT", tp), ("FACE", fp), ("AUDIO", ap)]:
        if pred is None:
            pred = {"dominant": "--", "emotions": {}}
        dom   = pred.get("dominant", "--")
        color = EMOTION_COLORS.get(dom, "#4A4844")
        top3  = sorted(pred.get("emotions", {}).items(),
                       key=lambda x: x[1], reverse=True)[:3]
        bars  = "".join(
            f"<div style='display:flex;gap:4px;font-family:Space Mono,monospace;"
            f"font-size:9px;color:#8A8278;margin:2px 0'>"
            f"<span style='color:{EMOTION_COLORS.get(e,'#4A4844')}'>{e}</span>"
            f"<span>{p:.2f}</span></div>"
            for e, p in top3
        )
        html += f"""
<div>
  <span style="font-family:Space Mono,monospace;font-size:10px;
               color:#8A8278;display:block;margin-bottom:4px">{label}</span>
  <span style="font-family:'Bebas Neue',sans-serif;font-size:24px;
               color:{color};display:block">{dom.upper()}</span>
  {bars}
</div>"""
    html += "</div>"
    return html


def _latency_md(tp, fp, ap):
    tl = f"{tp['latency_ms']:.0f}ms" if tp else "--"
    fl = f"{fp['latency_ms']:.0f}ms" if fp else "--"
    al = f"{ap['latency_ms']:.0f}ms" if ap else "--"
    return (
        f"<span style='font-family:Space Mono,monospace;font-size:10px;color:#4A4844'>"
        f"TEXT {tl}  /  FACE {fl}  /  AUDIO {al}</span>"
    )


def _run_fusion(tp, fp, ap, tw, fw, aw):
    """Run fusion and return all UI updates."""
    _tp = tp or predict_text("")
    _fp = fp or predict_face(None)
    _ap = ap or predict_audio(None)
    result = fuse(_tp, _fp, _ap, weights=(tw, fw, aw))
    _append_history(result, "FUSED")
    return (
        _dominant_html(result["dominant"], result["confidence"]),
        _bars_html(result["emotions"]),
        _latency_md(_tp, _fp, _ap),
        _weights_html(tw, fw, aw),
        _modality_html(_tp, _fp, _ap),
        _session_stats_html(),
        result,        # tp passthrough if unchanged
    )


# ── Tab 1 callbacks ───────────────────────────────────────────────────────────

def on_analyze_text(text, fp, ap, tw, fw, aw):
    tp = predict_text(text)
    result = fuse(tp, fp or predict_face(None),
                  ap or predict_audio(None), weights=(tw, fw, aw))
    _append_history(result, "TEXT")
    return (
        tp,
        _dominant_html(result["dominant"], result["confidence"]),
        _bars_html(result["emotions"]),
        _latency_md(tp, fp, ap),
        _weights_html(tw, fw, aw),
        _modality_html(tp, fp, ap),
        _session_stats_html(),
    )


def on_analyze_face(frame, tp, ap, tw, fw, aw):
    if frame is None:
        return (
            None,
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
        )
    fp = predict_face(frame)
    result = fuse(tp or predict_text(""), fp,
                  ap or predict_audio(None), weights=(tw, fw, aw))
    _append_history(result, "FACE")
    return (
        fp,
        _dominant_html(result["dominant"], result["confidence"]),
        _bars_html(result["emotions"]),
        _latency_md(tp, fp, ap),
        _weights_html(tw, fw, aw),
        _modality_html(tp, fp, ap),
        _session_stats_html(),
    )


def on_analyze_audio(audio, tp, fp, tw, fw, aw):
    if audio is None:
        return (
            None,
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
        )
    sr, data = audio
    ap = predict_audio(data.astype(np.float32) / 32768.0
                       if data.dtype == np.int16 else data.astype(np.float32),
                       sample_rate=sr)
    result = fuse(tp or predict_text(""), fp or predict_face(None),
                  ap, weights=(tw, fw, aw))
    _append_history(result, "AUDIO")
    return (
        ap,
        _dominant_html(result["dominant"], result["confidence"]),
        _bars_html(result["emotions"]),
        _latency_md(tp, fp, ap),
        _weights_html(tw, fw, aw),
        _modality_html(tp, fp, ap),
        _session_stats_html(),
    )


def on_fuse(tp, fp, ap, tw, fw, aw):
    _tp = tp or predict_text("")
    _fp = fp or predict_face(None)
    _ap = ap or predict_audio(None)
    result = fuse(_tp, _fp, _ap, weights=(tw, fw, aw))
    _append_history(result, "FUSED")
    return (
        _dominant_html(result["dominant"], result["confidence"]),
        _bars_html(result["emotions"]),
        _latency_md(_tp, _fp, _ap),
        _weights_html(tw, fw, aw),
        _modality_html(_tp, _fp, _ap),
        _session_stats_html(),
    )


# ── Tab 2 callback ────────────────────────────────────────────────────────────
def on_text2(text):
    tp = predict_text(text)
    _append_history(tp, "TEXT")
    return (
        _dominant_html(tp["dominant"], tp["confidence"]),
        _bars_html(tp["emotions"]),
    )


# ── Tab 3 callback ────────────────────────────────────────────────────────────
def on_audio3(audio):
    if audio is None:
        return gr.update(), gr.update()
    sr, data = audio
    ap = predict_audio(
        data.astype(np.float32) / 32768.0
        if data.dtype == np.int16 else data.astype(np.float32),
        sample_rate=sr,
    )
    _append_history(ap, "AUDIO")
    return (
        _dominant_html(ap["dominant"], ap["confidence"]),
        _bars_html(ap["emotions"]),
    )


# ── Wire callbacks ────────────────────────────────────────────────────────────
_live_outputs = [
    dominant_display, confidence_bars,
    latency_info, weight_display,
    modality_split, session_log,
]

analyze_text_btn.click(
    on_analyze_text,
    inputs=[text_input, face_pred_state, audio_pred_state,
            text_w_slider, face_w_slider, audio_w_slider],
    outputs=[txt_pred_state] + _live_outputs,
)

analyze_face_btn.click(
    on_analyze_face,
    inputs=[webcam_input, txt_pred_state, audio_pred_state,
            text_w_slider, face_w_slider, audio_w_slider],
    outputs=[face_pred_state] + _live_outputs,
)

analyze_audio_btn.click(
    on_analyze_audio,
    inputs=[audio_input, txt_pred_state, face_pred_state,
            text_w_slider, face_w_slider, audio_w_slider],
    outputs=[audio_pred_state] + _live_outputs,
)

fuse_btn.click(
    on_fuse,
    inputs=[txt_pred_state, face_pred_state, audio_pred_state,
            text_w_slider, face_w_slider, audio_w_slider],
    outputs=_live_outputs,
)

# Weight sliders update the weight display live
for slider in [text_w_slider, face_w_slider, audio_w_slider]:
    slider.change(
        lambda tw, fw, aw: _weights_html(tw, fw, aw),
        inputs=[text_w_slider, face_w_slider, audio_w_slider],
        outputs=[weight_display],
    )

analyze_text_btn_2.click(
    on_text2,
    inputs=[text_input_2],
    outputs=[dominant_display_2, confidence_bars_2],
)

analyze_audio_btn_3.click(
    on_audio3,
    inputs=[audio_input_3],
    outputs=[dominant_display_3, confidence_bars_3],
)

refresh_btn.click(
    lambda: _session_stats_html(),
    outputs=[stats_display],
)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ECHOMIND — Multimodal Emotion Intelligence")
    print("=" * 60)
    print("  Loading models (this may take 30-60s on first run)...")
    preload_all()
    print("  Starting Gradio on http://localhost:7860")
    print("=" * 60 + "\n")

    demo.launch(
        server_port=7860,
        server_name="0.0.0.0",
        share=False,
        show_error=True,
    )


# ── Callback helpers ──────────────────────────────────────────────────────────

def _weights_html(tw, fw, aw):
    total = (tw + fw + aw) or 1.0
    tw_n, fw_n, aw_n = tw/total, fw/total, aw/total
    rows = ""
    for label, val, color in [("TEXT", tw_n, "#E8A030"), ("FACE", fw_n, "#00BFA0"), ("AUDIO", aw_n, "#8A8278")]:
        rows += (f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
                 f'<span style="font-family:Space Mono,monospace;font-size:11px;color:#8A8278;width:48px">{label}</span>'
                 f'<div style="flex:1;height:4px;background:#2D2B27"><div style="width:{val*100:.1f}%;height:100%;background:{color}"></div></div>'
                 f'<span style="font-family:Space Mono,monospace;font-size:11px;color:{color};width:36px;text-align:right">{val:.0%}</span></div>')
    warn = "" if abs(tw+fw+aw - 1.0) < 0.05 else '<div style="font-family:Space Mono,monospace;font-size:10px;color:#E8A030;margin-top:4px">AUTO-NORMALIZING</div>'
    return f"<div>{rows}{warn}</div>"


def _modality_html(tp, fp, ap):
    html = '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">'
    for label, pred in [("TEXT", tp), ("FACE", fp), ("AUDIO", ap)]:
        if pred is None:
            pred = {"dominant": "--", "emotions": {}}
        dom = pred.get("dominant", "--")
        color = EMOTION_COLORS.get(dom, "#4A4844")
        top3 = sorted(pred.get("emotions", {}).items(), key=lambda x: x[1], reverse=True)[:3]
        bars = "".join(
            f'<div style="display:flex;gap:4px;font-family:Space Mono,monospace;font-size:9px;color:#8A8278;margin:2px 0">'
            f'<span style="color:{EMOTION_COLORS.get(e,"#4A4844")}">{e}</span><span>{p:.2f}</span></div>'
            for e, p in top3
        )
        html += (f'<div><span style="font-family:Space Mono,monospace;font-size:10px;color:#8A8278;display:block;margin-bottom:4px">{label}</span>'
                 f'<span style="font-family:Bebas Neue,sans-serif;font-size:24px;color:{color};display:block">{dom.upper()}</span>{bars}</div>')
    return html + "</div>"


def _latency_md(tp, fp, ap):
    tl = f"{tp['latency_ms']:.0f}ms" if tp else "--"
    fl = f"{fp['latency_ms']:.0f}ms" if fp else "--"
    al = f"{ap['latency_ms']:.0f}ms" if ap else "--"
    return f"<span style='font-family:Space Mono,monospace;font-size:10px;color:#4A4844'>TEXT {tl}  /  FACE {fl}  /  AUDIO {al}</span>"


# ── Tab 1 callbacks ───────────────────────────────────────────────────────────

def on_analyze_text(text, fp, ap, tw, fw, aw):
    tp = predict_text(text)
    result = fuse(tp, fp or predict_face(None), ap or predict_audio(None), weights=(tw, fw, aw))
    _append_history(result, "TEXT")
    return (tp, _dominant_html(result["dominant"], result["confidence"]),
            _bars_html(result["emotions"]), _latency_md(tp, fp, ap),
            _weights_html(tw, fw, aw), _modality_html(tp, fp, ap), _session_stats_html())


def on_analyze_face(frame, tp, ap, tw, fw, aw):
    if frame is None:
        return (None, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    fp = predict_face(frame)
    result = fuse(tp or predict_text(""), fp, ap or predict_audio(None), weights=(tw, fw, aw))
    _append_history(result, "FACE")
    return (fp, _dominant_html(result["dominant"], result["confidence"]),
            _bars_html(result["emotions"]), _latency_md(tp, fp, ap),
            _weights_html(tw, fw, aw), _modality_html(tp, fp, ap), _session_stats_html())


def on_analyze_audio(audio, tp, fp, tw, fw, aw):
    if audio is None:
        return (None, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    sr, data = audio
    arr = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
    ap = predict_audio(arr, sample_rate=sr)
    result = fuse(tp or predict_text(""), fp or predict_face(None), ap, weights=(tw, fw, aw))
    _append_history(result, "AUDIO")
    return (ap, _dominant_html(result["dominant"], result["confidence"]),
            _bars_html(result["emotions"]), _latency_md(tp, fp, ap),
            _weights_html(tw, fw, aw), _modality_html(tp, fp, ap), _session_stats_html())


def on_fuse(tp, fp, ap, tw, fw, aw):
    _tp = tp or predict_text("")
    _fp = fp or predict_face(None)
    _ap = ap or predict_audio(None)
    result = fuse(_tp, _fp, _ap, weights=(tw, fw, aw))
    _append_history(result, "FUSED")
    return (_dominant_html(result["dominant"], result["confidence"]),
            _bars_html(result["emotions"]), _latency_md(_tp, _fp, _ap),
            _weights_html(tw, fw, aw), _modality_html(_tp, _fp, _ap), _session_stats_html())


def on_text2(text):
    tp = predict_text(text)
    _append_history(tp, "TEXT")
    return _dominant_html(tp["dominant"], tp["confidence"]), _bars_html(tp["emotions"])


def on_audio3(audio):
    if audio is None:
        return gr.update(), gr.update()
    sr, data = audio
    arr = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
    ap = predict_audio(arr, sample_rate=sr)
    _append_history(ap, "AUDIO")
    return _dominant_html(ap["dominant"], ap["confidence"]), _bars_html(ap["emotions"])


# ── Wire callbacks ────────────────────────────────────────────────────────────

_live_outputs = [dominant_display, confidence_bars, latency_info, weight_display, modality_split, session_log]

analyze_text_btn.click(on_analyze_text,
    inputs=[text_input, face_pred_state, audio_pred_state, text_w_slider, face_w_slider, audio_w_slider],
    outputs=[txt_pred_state] + _live_outputs)

analyze_face_btn.click(on_analyze_face,
    inputs=[webcam_input, txt_pred_state, audio_pred_state, text_w_slider, face_w_slider, audio_w_slider],
    outputs=[face_pred_state] + _live_outputs)

analyze_audio_btn.click(on_analyze_audio,
    inputs=[audio_input, txt_pred_state, face_pred_state, text_w_slider, face_w_slider, audio_w_slider],
    outputs=[audio_pred_state] + _live_outputs)

fuse_btn.click(on_fuse,
    inputs=[txt_pred_state, face_pred_state, audio_pred_state, text_w_slider, face_w_slider, audio_w_slider],
    outputs=_live_outputs)

for slider in [text_w_slider, face_w_slider, audio_w_slider]:
    slider.change(lambda tw, fw, aw: _weights_html(tw, fw, aw),
                  inputs=[text_w_slider, face_w_slider, audio_w_slider],
                  outputs=[weight_display])

analyze_text_btn_2.click(on_text2, inputs=[text_input_2], outputs=[dominant_display_2, confidence_bars_2])
analyze_audio_btn_3.click(on_audio3, inputs=[audio_input_3], outputs=[dominant_display_3, confidence_bars_3])
refresh_btn.click(lambda: _session_stats_html(), outputs=[stats_display])


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ECHOMIND -- Multimodal Emotion Intelligence")
    print("="*60)
    print("  Loading models...")
    preload_all()
    print("  Open http://localhost:7860 in your browser")
    print("="*60 + "\n")
    demo.launch(server_port=7860, server_name="127.0.0.1", share=False, show_error=True, inbrowser=True)
