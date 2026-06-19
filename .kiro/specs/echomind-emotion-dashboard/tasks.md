# Tasks: EchoMind Emotion AI Dashboard

## Task List

### Phase 1: Project Scaffold

- [ ] 1.1 Create the project directory structure with all required files and folders
  - `echomind/app.py`, `echomind/requirements.txt`, `echomind/README.md`
  - `echomind/frontend/__init__.py`, `echomind/frontend/styles.css`, `echomind/frontend/components.py`, `echomind/frontend/mock_models.py`
  - `echomind/utils/__init__.py`, `echomind/utils/visualizations.py`
  - `echomind/assets/architecture_description.txt`
  - **Requirement**: 21.1

- [ ] 1.2 Populate `requirements.txt` with pinned dependencies
  - Include: `gradio>=4.0,<5.0`, `plotly>=5.0,<6.0`, `numpy`, `opencv-python`, `Pillow`
  - **Requirement**: 21.2

---

### Phase 2: Mock Models (`frontend/mock_models.py`)

- [ ] 2.1 Define `EMOTIONS` constant and `PredictionResult` / `HistoryEntry` TypedDicts
  - **Requirement**: 2.1, 3.1, 4.1, 5.1

- [ ] 2.2 Implement `predict_text(text: str) -> dict`
  - Keyword-biased random distribution over EMOTIONS
  - Returns valid PredictionResult shape (emotions, dominant, confidence, latency_ms)
  - Logs `[DEMO MODE]` on first call
  - **Requirement**: 2.1–2.6

- [ ] 2.3 Implement `predict_face(frame_array: np.ndarray | None) -> dict`
  - Sine-wave drift using `time.time()` for smooth variation
  - Handles `None` input gracefully (returns neutral-dominant result)
  - Logs `[DEMO MODE]` on first call
  - **Requirement**: 3.1–3.6

- [ ] 2.4 Implement `predict_audio(audio_data: tuple | None) -> dict`
  - Realistic random distribution over EMOTIONS
  - Handles `None` input gracefully (returns neutral-dominant result)
  - Logs `[DEMO MODE]` on first call
  - **Requirement**: 4.1–4.5

- [ ] 2.5 Implement `fuse_predictions(text_pred, face_pred, audio_pred, weights=(0.4, 0.35, 0.25)) -> dict`
  - Weighted late fusion with weight normalization
  - Returns valid PredictionResult shape
  - **Requirement**: 5.1–5.6

---

### Phase 3: Visualizations (`utils/visualizations.py`)

- [ ] 3.1 Define `EMOTION_COLORS` dict and shared chart theme helpers
  - Dark theme: `paper_bgcolor="rgba(0,0,0,0)"`, `plot_bgcolor="rgba(0,0,0,0)"`, font color `#e8e8f0`, gridline `#2d2f4a`
  - DEMO MODE annotation helper (bottom-right, italic)
  - **Requirement**: 6.3–6.5, 7.3–7.5, 8.4–8.5, 9.4–9.5, 10.3–10.5

- [ ] 3.2 Implement `make_radar_chart(emotions: dict, title: str = "") -> go.Figure`
  - Polar/scatterpolar trace, fixed height 320px
  - **Requirement**: 6.1–6.5

- [ ] 3.3 Implement `make_confidence_bars(emotions: dict) -> go.Figure`
  - Horizontal bar chart sorted descending, fixed height 280px
  - **Requirement**: 7.1–7.5

- [ ] 3.4 Implement `make_emotion_timeline(history: list[dict], window_seconds: int = 30) -> go.Figure`
  - Line chart of dominant emotion scores, fixed height 220px
  - Filters to `window_seconds` window; handles empty history gracefully
  - **Requirement**: 8.1–8.6

- [ ] 3.5 Implement `make_fusion_donut(fused: dict) -> go.Figure`
  - Donut pie chart with center annotation (dominant label + confidence %), fixed height 260px
  - **Requirement**: 9.1–9.5

- [ ] 3.6 Implement `make_modality_comparison(text_pred, face_pred, audio_pred) -> go.Figure`
  - Grouped bar chart with 3 traces × 7 emotions, fixed height 240px
  - **Requirement**: 10.1–10.5

---

### Phase 4: HTML Components (`frontend/components.py`)

- [ ] 4.1 Define `EMOTION_COLORS` and `EMOTION_EMOJIS` constants (import or redefine)

- [ ] 4.2 Implement `dominant_emotion_html(emotion: str, confidence: float) -> str`
  - Large centered card with emoji, label, confidence %, color from EMOTION_COLORS
  - **Requirement**: 11.1–11.3

- [ ] 4.3 Implement `metric_card_html(label: str, value: str, color: str = "#7c6fcd") -> str`
  - Stat card with label, value, accent color applied to value text
  - **Requirement**: 12.1–12.3

- [ ] 4.4 Implement `status_indicator_html(detected: bool, label: str) -> str`
  - Animated green pulse dot when `detected=True`, static grey when `False`
  - **Requirement**: 13.1–13.2

---

### Phase 5: Stylesheet (`frontend/styles.css`)

- [ ] 5.1 Define CSS custom properties for the full color palette (DESIGN system)
  - **Requirement**: 14.2

- [ ] 5.2 Implement `.card`, `.metric-card`, `.emotion-badge`, `.status-dot` classes
  - Animated pulse keyframe for `.status-dot`
  - **Requirement**: 14.1

- [ ] 5.3 Add responsive breakpoint at 768px collapsing 3-column to 1-column
  - **Requirement**: 14.3

- [ ] 5.4 Add smooth hover/active transitions and custom scrollbar (dark track, purple thumb)
  - **Requirement**: 14.4–14.5

- [ ] 5.5 Override Gradio default backgrounds with transparent backgrounds for plot containers
  - **Requirement**: 14.6

---

### Phase 6: Gradio App (`app.py`)

- [ ] 6.1 Set up module-level `emotion_history: list[HistoryEntry] = []` and `MAX_HISTORY = 120`
  - Implement FIFO eviction helper function
  - **Requirement**: 15.1–15.2

- [ ] 6.2 Build "🎭 Live Analysis" tab
  - Text input + Analyze button → `predict_text` → `fuse_predictions` → update charts + history
  - Webcam stream with 800ms debounce → `predict_face` → `fuse_predictions` → update charts
  - Audio upload → `predict_audio` → `fuse_predictions` → update charts
  - Status indicators, dominant emotion card, radar, confidence bars, fusion donut, modality comparison
  - **Requirement**: 16.1–16.6

- [ ] 6.3 Build "📝 Deep Text Analysis" tab
  - Text input → `predict_text` → display radar, confidence bars, fusion donut
  - **Requirement**: 17.1–17.2

- [ ] 6.4 Build "🎵 Audio Analysis" tab
  - Audio input → `predict_audio` → display radar and confidence bars
  - **Requirement**: 18.1–18.2

- [ ] 6.5 Build "📊 Session Stats" tab
  - Four metric cards (total analyses, dominant emotion, avg confidence, duration)
  - Emotion timeline + session log
  - `gr.Timer(5)` tick handler recomputes and updates all components
  - Handles empty history gracefully
  - **Requirement**: 19.1–19.3

- [ ] 6.6 Build "ℹ️ About & Architecture" tab
  - Load content from `assets/architecture_description.txt`
  - Display system description, DEMO MODE explanation, architecture info
  - **Requirement**: 20.1

- [ ] 6.7 Wire `gr.Blocks(css=...)` with `frontend/styles.css`, bind to port 7860, add DEMO MODE banner
  - Log `[DEMO MODE]` to console on startup
  - **Requirement**: 1.1–1.4, 14.1

---

### Phase 7: Supporting Files

- [ ] 7.1 Write `assets/architecture_description.txt` describing the system architecture and DEMO MODE
  - **Requirement**: 20.1

- [ ] 7.2 Write `README.md` with setup instructions, file structure overview, and launch command
  - **Requirement**: 21.1

---

### Phase 8: Verification

- [ ] 8.1 Write property-based tests for mock models
  - Test Property 1 (PredictionResult shape), Property 2 (normalization), Property 3 (dominant=argmax), Property 4 (confidence=dominant score), Property 5 (weight normalization)
  - **Validates**: Properties 1–5

- [ ] 8.2 Write property-based tests for visualization functions
  - Test Property 6 (chart heights), Property 7 (transparent backgrounds), Property 8 (DEMO MODE annotation), Property 9 (confidence bars sorted), Property 10 (timeline window filtering), Property 11 (donut center annotation), Property 12 (modality comparison traces)
  - **Validates**: Properties 6–12

- [ ] 8.3 Write property-based tests for HTML components
  - Test Property 13 (dominant_emotion_html content), Property 14 (metric_card_html content), Property 15 (status indicator state)
  - **Validates**: Properties 13–15

- [ ] 8.4 Write property-based tests for history management
  - Test Property 16 (capacity + FIFO eviction), Property 17 (HistoryEntry field completeness)
  - **Validates**: Properties 16–17

- [ ] 8.5 Verify `python app.py` launches cleanly on port 7860 and all five tabs render without errors
  - **Requirement**: 1.1–1.4
