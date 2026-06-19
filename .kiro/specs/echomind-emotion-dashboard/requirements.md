# Requirements Document

## Introduction

EchoMind is a multimodal emotion AI dashboard that detects and visualizes emotions from text, webcam (facial expression), and audio inputs simultaneously. The frontend operates in DEMO MODE using mock model outputs, with all predictions clearly labeled as demo data. The system provides a production-quality UI built with Python 3.10+, Gradio 4.x, and Plotly 5.x, serving on port 7860 via `python app.py`. Real model integration is deferred to a future phase.

## Glossary

- **Dashboard**: The complete EchoMind Gradio web UI served at port 7860
- **Mock_Models**: The module `frontend/mock_models.py` providing deterministic mock emotion predictions
- **Visualizations**: The module `utils/visualizations.py` producing Plotly charts
- **Components**: The module `frontend/components.py` providing HTML string builders
- **PredictionResult**: The canonical dict shape returned by all modality predictors and the fusion function, containing `emotions`, `dominant`, `confidence`, and `latency_ms` fields
- **HistoryEntry**: A time-stamped emotion snapshot stored in `emotion_history`, containing `timestamp`, `emotions`, `dominant`, and `source` fields
- **Session**: A continuous usage period during which `emotion_history` accumulates up to 120 entries
- **Emotion_History**: The module-level list in `app.py` holding up to 120 `HistoryEntry` items (FIFO eviction)
- **Fusion**: Weighted late fusion combining text, face, and audio predictions into a single `PredictionResult`
- **DEMO_MODE**: The operational state of the entire frontend where all predictions come from mock models, clearly indicated in the UI
- **Text_Predictor**: The `predict_text` function in Mock_Models
- **Face_Predictor**: The `predict_face` function in Mock_Models
- **Audio_Predictor**: The `predict_audio` function in Mock_Models
- **Fuser**: The `fuse_predictions` function in Mock_Models
- **Radar_Chart**: The polar/radar Plotly figure produced by `make_radar_chart`
- **Confidence_Bars**: The horizontal bar Plotly figure produced by `make_confidence_bars`
- **Timeline**: The line chart Plotly figure produced by `make_emotion_timeline`
- **Fusion_Donut**: The donut Plotly figure produced by `make_fusion_donut`
- **Modality_Comparison**: The grouped bar Plotly figure produced by `make_modality_comparison`
- **EARS**: Easy Approach to Requirements Syntax — the structured requirement pattern used throughout this document

---

## Requirements

### Requirement 1: Application Entry Point and Launch

**User Story:** As a developer, I want to launch EchoMind with `python app.py`, so that the dashboard is immediately accessible at port 7860 without additional configuration.

#### Acceptance Criteria

1. THE Dashboard SHALL be launchable by executing `python app.py` from the project root
2. THE Dashboard SHALL bind to port 7860 on startup
3. THE Dashboard SHALL display a DEMO MODE banner or persistent indicator that is visible across all tabs
4. WHEN the application starts, THE Dashboard SHALL log a `[DEMO MODE]` message to the console

---

### Requirement 2: Mock Text Emotion Prediction

**User Story:** As a developer, I want the text predictor to return keyword-biased emotion distributions, so that text inputs produce contextually plausible demo predictions.

#### Acceptance Criteria

1. WHEN `predict_text` is called with a non-empty string, THE Text_Predictor SHALL return a PredictionResult with all seven EMOTIONS keys present in the `emotions` dict
2. WHEN `predict_text` is called, THE Text_Predictor SHALL return a PredictionResult where `emotions` values sum to approximately 1.0 (within ±0.01 tolerance)
3. WHEN `predict_text` is called, THE Text_Predictor SHALL return a PredictionResult where `dominant` equals the key with the highest value in `emotions`
4. WHEN `predict_text` is called, THE Text_Predictor SHALL return a PredictionResult where `confidence` equals the value of the `dominant` emotion
5. WHEN `predict_text` is called for the first time, THE Text_Predictor SHALL log a `[DEMO MODE]` message to the console
6. IF `predict_text` is called with an empty string, THEN THE Text_Predictor SHALL return a valid PredictionResult defaulting toward a neutral distribution

---

### Requirement 3: Mock Face Emotion Prediction

**User Story:** As a developer, I want the face predictor to produce smoothly varying emotion distributions using sine-wave drift, so that the webcam stream appears live and dynamic in demo mode.

#### Acceptance Criteria

1. WHEN `predict_face` is called with a NumPy ndarray, THE Face_Predictor SHALL return a PredictionResult with all seven EMOTIONS keys present in the `emotions` dict
2. WHEN `predict_face` is called, THE Face_Predictor SHALL return a PredictionResult where `emotions` values sum to approximately 1.0 (within ±0.01 tolerance)
3. WHEN `predict_face` is called, THE Face_Predictor SHALL use `time.time()` to drive a sine-wave drift so successive calls produce smoothly varying distributions
4. WHEN `predict_face` is called, THE Face_Predictor SHALL return a PredictionResult where `dominant` equals the key with the highest value in `emotions`
5. WHEN `predict_face` is called for the first time, THE Face_Predictor SHALL log a `[DEMO MODE]` message to the console
6. IF `predict_face` is called with a None frame, THEN THE Face_Predictor SHALL return a valid PredictionResult with a neutral-dominant distribution

---

### Requirement 4: Mock Audio Emotion Prediction

**User Story:** As a developer, I want the audio predictor to return realistic random emotion distributions, so that audio inputs produce plausible demo predictions.

#### Acceptance Criteria

1. WHEN `predict_audio` is called with valid audio data, THE Audio_Predictor SHALL return a PredictionResult with all seven EMOTIONS keys present in the `emotions` dict
2. WHEN `predict_audio` is called, THE Audio_Predictor SHALL return a PredictionResult where `emotions` values sum to approximately 1.0 (within ±0.01 tolerance)
3. WHEN `predict_audio` is called, THE Audio_Predictor SHALL return a PredictionResult where `dominant` equals the key with the highest value in `emotions`
4. WHEN `predict_audio` is called for the first time, THE Audio_Predictor SHALL log a `[DEMO MODE]` message to the console
5. IF `predict_audio` is called with None audio data, THEN THE Audio_Predictor SHALL return a valid PredictionResult with a neutral-dominant distribution

---

### Requirement 5: Weighted Late Fusion

**User Story:** As a developer, I want predictions from all three modalities to be combined via weighted late fusion, so that the dashboard presents a unified emotion assessment.

#### Acceptance Criteria

1. WHEN `fuse_predictions` is called with three PredictionResult dicts, THE Fuser SHALL return a single PredictionResult containing all seven EMOTIONS keys
2. WHEN `fuse_predictions` is called, THE Fuser SHALL apply default weights of (0.4, 0.35, 0.25) for text, face, and audio respectively
3. WHEN `fuse_predictions` is called, THE Fuser SHALL normalize the provided weights to sum to 1.0 before applying them
4. WHEN `fuse_predictions` is called, THE Fuser SHALL return a PredictionResult where the `emotions` values sum to approximately 1.0 (within ±0.01 tolerance)
5. WHEN `fuse_predictions` is called, THE Fuser SHALL return a PredictionResult where `dominant` equals the key with the highest value in the fused `emotions`
6. WHEN `fuse_predictions` is called, THE Fuser SHALL return a PredictionResult where `confidence` equals the value of the fused `dominant` emotion

---

### Requirement 6: Radar Chart Visualization

**User Story:** As a user, I want a radar chart showing the emotion distribution, so that I can quickly see the relative strength of each emotion at a glance.

#### Acceptance Criteria

1. WHEN `make_radar_chart` is called with an emotions dict, THE Radar_Chart SHALL be a Plotly Figure with a polar/scatterpolar trace covering all seven emotions
2. WHEN `make_radar_chart` is called, THE Radar_Chart SHALL have a fixed height of 320 pixels
3. WHEN `make_radar_chart` is called, THE Radar_Chart SHALL apply transparent backgrounds (`paper_bgcolor` and `plot_bgcolor` set to `"rgba(0,0,0,0)"`)
4. WHEN `make_radar_chart` is called, THE Radar_Chart SHALL include a subtle italic "DEMO MODE" annotation in the bottom-right corner
5. WHEN `make_radar_chart` is called, THE Radar_Chart SHALL use font color `#e8e8f0` and gridline color `#2d2f4a`

---

### Requirement 7: Confidence Bar Chart Visualization

**User Story:** As a user, I want a horizontal bar chart of emotion confidence scores sorted from highest to lowest, so that I can compare confidence levels across all emotions.

#### Acceptance Criteria

1. WHEN `make_confidence_bars` is called with an emotions dict, THE Confidence_Bars SHALL be a Plotly Figure with a horizontal bar trace covering all seven emotions sorted in descending order
2. WHEN `make_confidence_bars` is called, THE Confidence_Bars SHALL have a fixed height of 280 pixels
3. WHEN `make_confidence_bars` is called, THE Confidence_Bars SHALL apply transparent backgrounds (`paper_bgcolor` and `plot_bgcolor` set to `"rgba(0,0,0,0)"`)
4. WHEN `make_confidence_bars` is called, THE Confidence_Bars SHALL include a subtle italic "DEMO MODE" annotation in the bottom-right corner
5. WHEN `make_confidence_bars` is called, THE Confidence_Bars SHALL use font color `#e8e8f0` and gridline color `#2d2f4a`

---

### Requirement 8: Emotion Timeline Visualization

**User Story:** As a user, I want a line chart showing how dominant emotion scores change over the current session, so that I can observe emotion trends over time.

#### Acceptance Criteria

1. WHEN `make_emotion_timeline` is called with a history list, THE Timeline SHALL be a Plotly Figure with line traces for dominant emotions over time
2. WHEN `make_emotion_timeline` is called, THE Timeline SHALL have a fixed height of 220 pixels
3. WHEN `make_emotion_timeline` is called with a `window_seconds` parameter, THE Timeline SHALL display only entries within that time window
4. WHEN `make_emotion_timeline` is called, THE Timeline SHALL apply transparent backgrounds (`paper_bgcolor` and `plot_bgcolor` set to `"rgba(0,0,0,0)"`)
5. WHEN `make_emotion_timeline` is called, THE Timeline SHALL include a subtle italic "DEMO MODE" annotation in the bottom-right corner
6. IF `make_emotion_timeline` is called with an empty history list, THEN THE Timeline SHALL return a valid empty Plotly Figure without raising an error

---

### Requirement 9: Fusion Donut Chart Visualization

**User Story:** As a user, I want a donut chart displaying the dominant emotion and its confidence score at the center, so that I can instantly identify the primary emotion from the fused prediction.

#### Acceptance Criteria

1. WHEN `make_fusion_donut` is called with a fused PredictionResult, THE Fusion_Donut SHALL be a Plotly Figure with a donut (hole > 0) pie trace for all seven emotions
2. WHEN `make_fusion_donut` is called, THE Fusion_Donut SHALL annotate the dominant emotion label and confidence percentage at the center of the donut
3. WHEN `make_fusion_donut` is called, THE Fusion_Donut SHALL have a fixed height of 260 pixels
4. WHEN `make_fusion_donut` is called, THE Fusion_Donut SHALL apply transparent backgrounds (`paper_bgcolor` and `plot_bgcolor` set to `"rgba(0,0,0,0)"`)
5. WHEN `make_fusion_donut` is called, THE Fusion_Donut SHALL include a subtle italic "DEMO MODE" annotation in the bottom-right corner

---

### Requirement 10: Modality Comparison Chart Visualization

**User Story:** As a user, I want a grouped bar chart comparing emotion scores across all three modalities, so that I can see how text, face, and audio predictions differ.

#### Acceptance Criteria

1. WHEN `make_modality_comparison` is called with three PredictionResult dicts, THE Modality_Comparison SHALL be a Plotly Figure with three grouped bar traces (one per modality) across all seven emotions
2. WHEN `make_modality_comparison` is called, THE Modality_Comparison SHALL have a fixed height of 240 pixels
3. WHEN `make_modality_comparison` is called, THE Modality_Comparison SHALL apply transparent backgrounds (`paper_bgcolor` and `plot_bgcolor` set to `"rgba(0,0,0,0)"`)
4. WHEN `make_modality_comparison` is called, THE Modality_Comparison SHALL include a subtle italic "DEMO MODE" annotation in the bottom-right corner
5. WHEN `make_modality_comparison` is called, THE Modality_Comparison SHALL use font color `#e8e8f0` and gridline color `#2d2f4a`

---

### Requirement 11: Dominant Emotion HTML Component

**User Story:** As a user, I want a large centered card showing the current dominant emotion with its emoji, label, and confidence percentage, so that the primary emotion is immediately visible.

#### Acceptance Criteria

1. WHEN `dominant_emotion_html` is called with an emotion and confidence, THE Components SHALL return a valid HTML string containing the emotion's emoji, label, and confidence percentage
2. WHEN `dominant_emotion_html` is called, THE Components SHALL color-code the display using the emotion's entry in EMOTION_COLORS
3. WHEN `dominant_emotion_html` is called, THE Components SHALL center the display within the card

---

### Requirement 12: Metric Card HTML Component

**User Story:** As a developer, I want reusable metric cards for session statistics, so that the Session Stats tab can display key numbers in a consistent visual style.

#### Acceptance Criteria

1. WHEN `metric_card_html` is called with a label, value, and optional color, THE Components SHALL return a valid HTML string containing the label and value
2. WHEN `metric_card_html` is called without a color argument, THE Components SHALL default the accent color to `#7c6fcd`
3. WHEN `metric_card_html` is called, THE Components SHALL apply the accent color to the value text

---

### Requirement 13: Status Indicator HTML Component

**User Story:** As a user, I want animated status indicators showing whether each modality input is active, so that I can see at a glance which inputs are currently detected.

#### Acceptance Criteria

1. WHEN `status_indicator_html` is called with `detected=True`, THE Components SHALL return an HTML string with an animated green pulse dot and the provided label
2. WHEN `status_indicator_html` is called with `detected=False`, THE Components SHALL return an HTML string with a static grey dot and the provided label

---

### Requirement 14: Dark Theme Stylesheet

**User Story:** As a user, I want a consistent dark theme across all UI elements, so that the dashboard is visually cohesive and comfortable to use in low-light environments.

#### Acceptance Criteria

1. THE Dashboard SHALL load `frontend/styles.css` via `gr.Blocks(css=...)` on startup
2. THE Dashboard SHALL define CSS custom properties (variables) for the full color palette
3. THE Dashboard SHALL apply a responsive breakpoint at 768px that collapses 3-column layouts to 1-column
4. THE Dashboard SHALL apply smooth transitions on hover and active states for interactive elements
5. THE Dashboard SHALL render a custom scrollbar with a dark track and purple thumb
6. THE Dashboard SHALL override Gradio default backgrounds with transparent backgrounds for plot containers

---

### Requirement 15: Emotion History Management

**User Story:** As a developer, I want a bounded rolling history of emotion snapshots, so that session stats and timelines remain performant without unbounded memory growth.

#### Acceptance Criteria

1. THE Dashboard SHALL maintain `emotion_history` as a module-level list in `app.py` with a maximum capacity of 120 entries
2. WHEN a new HistoryEntry is appended and `emotion_history` has reached 120 entries, THE Dashboard SHALL evict the oldest entry (FIFO) before appending the new one
3. WHEN an analysis is performed via any modality, THE Dashboard SHALL append a HistoryEntry containing `timestamp`, `emotions`, `dominant`, and `source` fields

---

### Requirement 16: Live Analysis Tab

**User Story:** As a user, I want a Live Analysis tab that displays real-time emotion analysis from all three input channels simultaneously, so that I can explore multimodal emotion detection interactively.

#### Acceptance Criteria

1. THE Dashboard SHALL provide a "🎭 Live Analysis" tab containing text input, webcam stream, audio upload, dominant emotion card, status indicators, radar chart, confidence bars, fusion donut, and modality comparison chart
2. WHEN a user submits text input via the Analyze button, THE Dashboard SHALL call `predict_text`, `fuse_predictions`, append to history, and update all live charts
3. WHEN a webcam frame is received, THE Dashboard SHALL debounce face analysis at 800ms intervals and update charts on each processed frame
4. WHEN a webcam frame is processed, THE Dashboard SHALL call `predict_face`, `fuse_predictions`, append to history, and update all live charts
5. WHEN audio data is provided, THE Dashboard SHALL call `predict_audio`, `fuse_predictions`, append to history, and update all live charts
6. WHEN any analysis completes, THE Dashboard SHALL update the status indicators to reflect which modalities are currently active

---

### Requirement 17: Deep Text Analysis Tab

**User Story:** As a user, I want a dedicated Deep Text Analysis tab, so that I can focus on text-based emotion predictions with detailed charts.

#### Acceptance Criteria

1. THE Dashboard SHALL provide a "📝 Deep Text Analysis" tab containing a text input area and the radar chart, confidence bars, and fusion donut charts
2. WHEN text is submitted in the Deep Text Analysis tab, THE Dashboard SHALL call `predict_text` and display the radar chart, confidence bars, and fusion donut for the result

---

### Requirement 18: Audio Analysis Tab

**User Story:** As a user, I want a dedicated Audio Analysis tab, so that I can focus on audio-based emotion predictions.

#### Acceptance Criteria

1. THE Dashboard SHALL provide a "🎵 Audio Analysis" tab containing an audio input component and radar chart and confidence bars charts
2. WHEN audio is submitted in the Audio Analysis tab, THE Dashboard SHALL call `predict_audio` and display the radar chart and confidence bars for the result

---

### Requirement 19: Session Stats Tab

**User Story:** As a user, I want a Session Stats tab that summarizes my session's emotion activity, so that I can review aggregated patterns from the current session.

#### Acceptance Criteria

1. THE Dashboard SHALL provide a "📊 Session Stats" tab containing four metric cards (total analyses, dominant emotion, average confidence, session duration), an emotion timeline, and a session log
2. WHEN a `gr.Timer` fires every 5 seconds, THE Dashboard SHALL recompute and update all four metric cards, the emotion timeline, and the session log from `emotion_history`
3. IF `emotion_history` is empty when the timer fires, THEN THE Dashboard SHALL display zeroed or placeholder values in the metric cards without raising an error

---

### Requirement 20: About & Architecture Tab

**User Story:** As a user, I want an About & Architecture tab explaining the system design, so that I understand how EchoMind works and what the demo mode entails.

#### Acceptance Criteria

1. THE Dashboard SHALL provide an "ℹ️ About & Architecture" tab containing a description of the system, an explanation of DEMO MODE, and architecture information sourced from `assets/architecture_description.txt`

---

### Requirement 21: File Structure and Module Organization

**User Story:** As a developer, I want the codebase organized according to the specified file structure, so that modules are isolated and real model integration can replace mock modules independently.

#### Acceptance Criteria

1. THE Dashboard SHALL organize code into `app.py`, `frontend/__init__.py`, `frontend/styles.css`, `frontend/components.py`, `frontend/mock_models.py`, `utils/__init__.py`, `utils/visualizations.py`, `assets/architecture_description.txt`, `requirements.txt`, and `README.md`
2. THE Dashboard SHALL declare all Python dependencies in `requirements.txt` with pinned or bounded versions
3. THE Mock_Models module SHALL be isolated such that each `predict_*` function can be replaced independently without modifying other modules
