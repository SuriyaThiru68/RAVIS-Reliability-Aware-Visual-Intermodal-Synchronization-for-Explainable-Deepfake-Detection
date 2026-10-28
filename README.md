# RAVIS
### Reliability-Aware Visual-Intermodal Synchronization for Explainable Deepfake Detection

> Final Year Project — AI-powered deepfake video detection using multimodal audio-visual synchronization analysis, Random Forest classification, SHAP attribution, and conformal prediction.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Processing Pipeline](#processing-pipeline)
4. [Project Structure](#project-structure)
5. [Tech Stack](#tech-stack)
6. [Getting Started](#getting-started)
7. [How to Run](#how-to-run)
8. [API Reference](#api-reference)
9. [Detection Modes](#detection-modes)
10. [ML Model](#ml-model)
11. [Results Dashboard](#results-dashboard)
12. [Screenshots](#screenshots)

---

## Overview

RAVIS is a browser-based deepfake detection system that analyses whether a video has been synthetically manipulated. It works by measuring the coherence between a speaker's lip movements and their speech audio — a relationship that deepfake synthesis tools consistently break.

The system implements an 8-stage forensic pipeline modelled on the RAVIS paper, combining signal-processing methods (cross-correlation, FastDTW, Wasserstein distance) with machine learning (Random Forest + SHAP + conformal prediction) and a live in-browser camera mode powered by MediaPipe.

---

## Key Features

| Category | Features |
|---|---|
| **Upload Detection** | Drag-and-drop video upload · Demo presets (Fake / Real) · 9-stage async pipeline with live progress |
| **Live Camera** | Real-time webcam recording · MediaPipe lip mesh overlay · Live MAR / RMS / sync rolling charts |
| **Forensic Analysis** | Cross-correlation · FastDTW · Wasserstein distance · Lip-velocity cross-correlation · Whisper phoneme sync |
| **Emotion Consistency** | FER-based per-frame facial emotion · Prosody audio emotion · Cross-modal valence-arousal disagreement score |
| **Explainability** | Per-sample SHAP attribution · Conformal prediction sets (90% coverage) · Plain-English narrative |
| **Reliability** | SNR estimation · Face tracking coverage · Audio coverage · Duration grading (HIGH / MEDIUM / LOW) |
| **Reporting** | Downloadable PDF report · Server-side SHA-256 integrity hash · Video thumbnails embedded in report |
| **History** | Session history table · Re-view any past result |

---

## Processing Pipeline

The 8-stage pipeline from the RAVIS paper is fully implemented:

```
┌─────┬─────────────────────────┬──────────────────────────────────────────────────────────┐
│  #  │  Stage                  │  Implementation                                          │
├─────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
│  1  │  Preprocessing          │  OpenCV frame extraction, ffmpeg audio (16 kHz WAV),     │
│     │                         │  resolution + frame-rate metadata preserved               │
├─────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
│  2  │  Face / Lip Kinematics  │  MediaPipe FaceMesh (478 landmarks), MAR per frame,      │
│     │                         │  lip velocity, head pose (yaw / pitch), face area ratio  │
├─────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
│  3  │  Acoustic Extraction    │  librosa: 13-coeff MFCC + delta, 64-band Mel, RMS,       │
│     │                         │  ZCR, spectral centroid/rolloff/bandwidth                 │
│     │                         │  + SNR estimation (noise-floor percentile)                │
│     │                         │  + Whisper word-level speech timestamps (optional)        │
├─────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
│  4  │  Synchronization        │  Cross-correlation (±0.75 s lag window)                  │
│     │                         │  FastDTW (falls back to exact DTW)                       │
│     │                         │  Wasserstein / Earth Mover's Distance                    │
│     │                         │  Lip-velocity cross-correlation                           │
│     │                         │  Whisper-aligned phoneme sync score                      │
├─────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
│  5  │  Emotion Consistency    │  FER facial emotion per frame (7 labels)                 │
│     │                         │  Prosody audio emotion (valence / arousal)                │
│     │                         │  Cross-modal disagreement score                          │
├─────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
│  6  │  Reliability Estimation │  Face detection rate, audio coverage, duration,          │
│     │                         │  SNR, emotion consistency → HIGH / MEDIUM / LOW grade    │
├─────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
│  7  │  Classification         │  Random Forest (200 trees, 74 features)                  │
│     │                         │  Physics-evidence fusion (45 / 55 blend)                 │
│     │                         │  SHAP TreeExplainer — per-sample feature attribution     │
│     │                         │  Conformal prediction set at 90 % coverage               │
├─────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
│  8  │  Reporting              │  Server-side SHA-256 file integrity hash                 │
│     │                         │  Downloadable PDF with video thumbnails + sync chart     │
│     │                         │  Dashboard verdict + per-second timeline                 │
└─────┴─────────────────────────┴──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Final Year Project/
│
├── backend/
│   ├── app.py                    # Flask server — API routes + pipeline orchestrator
│   ├── report_pdf.py             # ReportLab PDF report generator
│   ├── models/
│   │   ├── rf_model.pkl          # Trained Random Forest (joblib)
│   │   └── feature_names.pkl     # Feature key list
│   └── pipeline/
│       ├── __init__.py
│       ├── video_processor.py    # Stage 1 — frame extraction, audio extraction
│       ├── face_detector.py      # Stage 2 — MediaPipe FaceMesh, MAR, lip velocity
│       ├── audio_features.py     # Stage 3 — MFCC, Mel, SNR, Whisper timestamps
│       ├── visual_features.py    # Stage 2 — aggregate visual feature stats
│       ├── sync_analyzer.py      # Stage 4 — xcorr, FastDTW, Wasserstein, Whisper sync
│       ├── emotion_analyzer.py   # Stage 5 — FER facial emotion + audio emotion
│       ├── classifier.py         # Stage 7 — RF, SHAP, conformal prediction
│       └── explainer.py          # Stage 6/8 — reliability, explanation, format_results
│
├── frontend/
│   ├── index.html                # Single-page application shell
│   ├── css/
│   │   └── style.css             # Swiss design system (white + electric blue #3533FF)
│   └── js/
│       ├── app.js                # Upload flow, result rendering, history, PDF download
│       └── realtime.js           # Live camera recording + real-time metrics
│
├── train/                        # Model training scripts
├── tools/                        # Utility scripts
├── run.py                        # Entry point — starts Flask server
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Tech Stack

### Backend

| Library | Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| Flask | 3.0.3 | Web framework + REST API |
| Flask-CORS | 4.0.1 | Cross-origin support |
| OpenCV | 4.10 | Frame extraction, image processing |
| MediaPipe | 0.10 | FaceMesh 478-point landmark detection |
| librosa | 0.10 | MFCC, Mel spectrogram, RMS, ZCR |
| scipy | 1.13 | Cross-correlation, cosine similarity |
| scikit-learn | 1.5 | Random Forest classifier |
| joblib | 1.4 | Model serialization |
| numpy | 1.26 | Numerical computing |
| ReportLab | — | PDF report generation |
| openai-whisper | — | Speech timestamps *(optional)* |
| fastdtw | — | Fast Dynamic Time Warping *(optional)* |
| shap | — | SHAP feature attribution *(optional)* |
| fer | — | Facial emotion recognition *(optional)* |
| moviepy | 1.0 | Audio extraction fallback |

### Frontend

| Technology | Purpose |
|---|---|
| HTML5 / CSS3 / Vanilla JS | No framework — zero dependencies |
| MediaPipe FaceMesh (CDN) | Browser-side live lip tracking (478 landmarks) |
| MediaRecorder API | Webcam + microphone recording |
| Web Audio API | Browser-side RMS energy analysis |
| Canvas API | Lip mesh overlay rendering |

---

## Getting Started

### Prerequisites

- **Python 3.11** (tested; 3.10+ should work)
- **Git** (to clone the repo)
- **ffmpeg** — bundled automatically via `imageio-ffmpeg` (no manual install needed)

### 1. Clone / open the project

```powershell
Set-Location 'e:\Final Year Project'
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install optional packages (for full pipeline)

```powershell
# FastDTW — faster DTW for long clips
pip install fastdtw

# SHAP — per-sample feature attribution
pip install shap

# Facial emotion recognition
pip install fer

# Whisper — speech timestamps (downloads ~75 MB model on first use)
pip install openai-whisper
```

> All optional packages degrade gracefully — the pipeline runs without them, just with reduced features.

---

## How to Run

```powershell
Set-Location 'e:\Final Year Project'
.\.venv\Scripts\python.exe -B run.py
```

Then open your browser at:

```
http://localhost:5000
```

The server will:
- Auto-train the Random Forest model if `rf_model.pkl` is missing
- Serve the frontend from `/frontend/`
- Handle all API requests on port 5000

---

## API Reference

### `POST /api/analyze`
Upload a video file for analysis.

**Request:** `multipart/form-data` with field `video`

**Response:**
```json
{ "job_id": "uuid-string" }
```

---

### `GET /api/status/{job_id}`
Poll the status of a running analysis job.

**Response:**
```json
{
  "job_id": "...",
  "status": "running",
  "progress": 65,
  "stage": "Computing audio-visual synchronization",
  "result": null,
  "error": null
}
```

`status` values: `queued` → `running` → `done` / `failed`

---

### `POST /api/report`
Generate and download a PDF forensic report.

**Request body:**
```json
{
  "result": { ... },
  "filename": "video.mp4",
  "sha256": "abc123...",
  "elapsed": 12.3,
  "job_id": "uuid-string"
}
```

**Response:** `application/pdf` file download

---

### `POST /api/realtime_score`
Get a backend RF verdict from pre-computed live camera features.

**Request body:**
```json
{
  "mar_mean": 0.18,
  "rms_mean": 0.042,
  "sync_score": 0.61,
  "xcorr_peak": 0.44,
  "lag_ms": 28.0
}
```

---

### `GET /api/health`
Health check.

**Response:** `{ "status": "ok" }`

---

## Detection Modes

### Mode 1 — Upload Video

1. Drag and drop (or browse) a video file — MP4, AVI, MOV, MKV, or WebM up to **500 MB**
2. Click **Analyze Video**
3. Watch the live progress bar as each pipeline stage completes
4. View the full results dashboard with verdict, metrics, timeline, and explanation
5. Download a PDF forensic report

### Mode 2 — Live Camera

1. Open the **Live Camera** tab
2. Click **Start Recording** — grant webcam + microphone permissions
3. Speak naturally for at least 3–5 seconds
4. Watch the live lip mesh overlay, MAR chart, audio level, and sync score update in real time
5. Click **Stop & Analyze** — the recording is sent through the same full pipeline
6. Results appear below the camera view

---

## ML Model

| Property | Value |
|---|---|
| Algorithm | Random Forest |
| Trees | 200 |
| Max depth | 15 |
| Features | 74 (audio MFCC + visual MAR + sync metrics) |
| Training data | Synthetic feature distributions (1 600 samples) |
| Test AUC | ~0.97 on held-out synthetic test set |
| Model file | `backend/models/rf_model.pkl` |
| Explainability | SHAP TreeExplainer (per-sample attribution) |
| Uncertainty | Conformal prediction sets at 90 % coverage |
| Fusion | 45 % learned model + 55 % physics evidence |

The classifier is **fused** with a physics-based evidence score derived directly from the measured synchronization values (cross-correlation, DTW score, audio-lip lag, suspicious-window fraction). This anchors decisions to the actual measurements rather than solely to the training distribution.

---

## Results Dashboard

Each analysis produces:

- **Verdict banner** — `DEEPFAKE` / `AUTHENTIC` / `INCONCLUSIVE` with confidence %
- **Core measurements table** — lip-audio sync score, A/V offset (ms), cross-correlation peak, DTW distance, face tracking coverage
- **Plain-English explanation** — narrative summary of the detection reasoning
- **Per-second timeline** — green / yellow / red colour-coded segments for each second of the video
- **Suspicious moments list** — flagged timestamps with individual sync scores
- **3D face landmarker** — live 478-point mesh with head pose (yaw / pitch / roll) and kinetic energy
- **A/V synchronisation chart** — audio energy, lip MAR, and sync curve plotted over time
- **Forensic evidence panel** — 6 evidence cards (audio forensics, pixel forensics, lip sync kinematics, temporal consistency, motion irregularity, compression artifacts)
- **Reasoning trace** — numbered decision steps
- **Chain of custody** — SHA-256 hash, job ID, timestamp, model identity

---

## Reliability Grades

| Grade | Score | Meaning |
|---|---|---|
| HIGH | ≥ 0.85 | Strong evidence — result is trustworthy |
| MEDIUM | 0.60 – 0.84 | Adequate evidence — minor caveats |
| LOW | 0.35 – 0.59 | Weak evidence — interpret with caution |
| VERY LOW | < 0.35 | Insufficient evidence — result withheld or unreliable |

Reliability is computed from face detection rate, audio coverage, video duration, SNR, and emotion consistency.

---

## Notes

- The system is designed for **decision support**, not final attribution. A trained examiner should correlate results with source provenance and device metadata before formal forensic use.
- Deepfake detection is an adversarial domain — no detector achieves 100 % accuracy on all manipulation types.
- The Whisper, FastDTW, SHAP, and FER packages are optional. All degrade gracefully if not installed.

---

*RAVIS — Reliability-Aware Visual-Intermodal Synchronization for Explainable Deepfake Detection*
