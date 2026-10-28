# Multi-Scale Synchronization Analysis for Deepfake Detection

## 1. Project Goal

Build a deepfake detection system that analyzes audio-video synchronization at three levels:

1. Frame level
2. Word level
3. Video level

The system should detect whether a video is real or manipulated by comparing mouth motion, speech timing, audio features, and sequence consistency.

---

## 2. Main Idea

The project uses a multi-scale synchronization framework:

- Frame-level analysis checks temporal consistency in each frame.
- Word-level analysis checks whether spoken words align with lip movements over time.
- Video-level analysis compares the whole signal sequence using correlation, DTW, and confidence scoring.

This is the central idea shown in the uploaded concept:

- Multi-Scale Synchronization Analysis with three levels
- Frame Level
- Word Level
- Video Level

---

## 3. System Architecture

### 3.1 Input and Preprocessing

Input:
- Video file (MP4 / AVI / MOV)
- Audio extracted from the video

Preprocessing steps:
- Extract frames from the video
- Detect face landmarks using MediaPipe
- Track mouth region and lip movement
- Extract audio waveform and MFCC features using Librosa
- Normalize timestamps and frame rates

### 3.2 Feature Extraction

#### A. Visual features
- Face detection rate
- Mouth region movement
- Lip velocity
- Lip activity ratio
- Facial landmarks and expressions
- Head motion patterns

#### B. Audio features
- MFCC mean and standard deviation
- Mel features
- Spectral centroid
- Zero-crossing rate
- RMS energy
- Silence ratio
- Speech activity and dynamic range

#### C. Synchronization features
- Cross-correlation score
- Audio-lip lag
- Dynamic Time Warping (DTW) distance
- DTW score
- Cosine similarity between speech and lip sequences
- Alignment score
- Suspicious window fraction

---

## 4. Three-Level Implementation Plan

## 4.1 Frame-Level Synchronization

Purpose:
- Measure whether each frame shows correct audio-video alignment.

Implementation:
- Extract frames at fixed intervals
- Detect face and mouth landmarks in each frame
- Estimate lip opening and lip motion energy
- Compare lip motion signal to short-window audio energy envelopes
- Compute frame-wise sync score for each timestamp

Key metrics:
- Lip opening amplitude
- Mouth width/height ratio
- Face motion variance
- Audio energy envelope
- Short-term alignment error

Output:
- Frame-wise alignment map
- Frame-level sync confidence
- Suspicious regions where lip motion is inconsistent with audio

---

## 4.2 Word-Level Synchronization

Purpose:
- Detect whether spoken words match the mouth movement at the word level.

Implementation:
- Convert audio to phoneme/speech segments
- Detect speech regions and word boundaries
- Map each segment to corresponding mouth motion timing
- Compare audio features to lip movement for each spoken segment

Key metrics:
- Word onset timing
- Mouth opening duration
- Speech-to-lip mapping score
- Word boundary mismatch score
- Temporal lag per word

Output:
- Word-by-word synchronization score
- Mismatched words or syllables
- Time-shift detection

---

## 4.3 Video-Level Synchronization

Purpose:
- Evaluate the entire video sequence as one temporal unit.

Implementation:
- Aggregate all frame-level and word-level scores
- Compute global audio-video consistency metrics
- Evaluate long-range mismatch patterns
- Detect unnatural motion sequences and discontinuities

Key metrics:
- Cross-correlation score
- DTW distance and score
- Mean alignment score
- Overall sync score
- Suspicious window fraction
- Long-range temporal inconsistency

Output:
- Final video-level sync score
- Deepfake probability
- Confidence value

---

## 5. Model Design

The current project already uses a Random Forest classifier:

- Training file: `train/train_model.py`
- Model loader: `backend/pipeline/classifier.py`
- Trained model: `backend/models/rf_model.pkl`

This model takes a combined feature vector built from:
- audio features
- visual features
- synchronization features

The Random Forest predicts:
- Real
- Deepfake

### Suggested implementation flow

1. Extract features
2. Build feature vector
3. Run synchronization analysis
4. Feed into Random Forest classifier
5. Return verdict, confidence, and explanation

---

## 6. Recommended Code Structure

### Backend

- `backend/app.py`
  - Flask API entry point
  - handles upload and prediction request

- `backend/pipeline/audio_features.py`
  - MFCC, RMS, spectral, silence, speech activity

- `backend/pipeline/visual_features.py`
  - face detection, mouth movement, lip trajectory

- `backend/pipeline/sync_analyzer.py`
  - cross-correlation, DTW, alignment, lag analysis

- `backend/pipeline/classifier.py`
  - Random Forest model loading and prediction

- `backend/pipeline/explainer.py`
  - model explanation and top features

### Training

- `train/train_model.py`
  - synthetic data generation
  - model training
  - save RF model

### Frontend

- `frontend/index.html`
- `frontend/css/style.css`
- `frontend/js/app.js`

---

## 7. Step-by-Step Implementation Tasks

### Phase 1: Environment and baseline setup
- [ ] Activate virtual environment
- [ ] Install requirements
- [ ] Confirm project runs with `python run.py`
- [ ] Validate model file exists
- [ ] Test startup without errors

### Phase 2: Frame-level module
- [ ] Implement frame extraction
- [ ] Detect face and mouth landmarks
- [ ] Estimate mouth openness / lip motion
- [ ] Extract short-term features per frame
- [ ] Compute frame-sync score

### Phase 3: Word-level module
- [ ] Separate audio into speech segments
- [ ] Detect word boundaries
- [ ] Align words with mouth motion
- [ ] Compute word-level mismatch metrics
- [ ] Store per-word synchronization results

### Phase 4: Video-level module
- [ ] Aggregate frame-level and word-level scores
- [ ] Compute cross-correlation and DTW metrics
- [ ] Create global sync score
- [ ] Detect suspicious windows and anomalies

### Phase 5: Feature fusion and classification
- [ ] Build final combined feature vector
- [ ] Confirm all required keys match the trained model
- [ ] Run Random Forest prediction
- [ ] Return deepfake probability and verdict

### Phase 6: Explainability and reporting
- [ ] Show top feature importance
- [ ] Display synchronization evidence
- [ ] Generate simple explanation
- [ ] Present results in UI

### Phase 7: Testing and validation
- [ ] Test on real videos
- [ ] Test on deepfake inputs
- [ ] Check confidence thresholds
- [ ] Improve model if needed
- [ ] Final validation report

---

## 8. Implementation Priority

### Priority 1: Core working pipeline
- Video upload
- Audio extraction
- Face and lip detection
- Feature vector creation
- Random Forest prediction

### Priority 2: Synchronization quality
- Frame-level consistency
- Word-level alignment
- Video-level aggregation

### Priority 3: Explanation and final UI
- Top features
- Evidence summary
- Human-readable output

---

## 9. Expected Output

The final app should produce:

- Prediction: Real / Deepfake
- Deepfake probability
- Confidence score
- Top contributing features
- Synchronization explanation
- Visual and audio evidence summary

---

## 10. Acceptance Criteria

The feature is complete when:

- A video can be uploaded successfully
- Audio and visual features are extracted
- Frame, word, and video synchronization are computed
- Final prediction is generated by the classifier
- The user can see a clear verdict and evidence

---

## 11. Run Command

From the project root:

```powershell
Set-Location "e:\Final Year Project"
.\.venv\Scripts\python.exe run.py
```

---

## 12. Summary

This project is best understood as a multi-scale synchronization-based deepfake detector. The key novelty is not only using visual or audio features individually, but combining them across:

- frame level
- word level
- video level

This is how the system can detect subtle lip-sync manipulation and temporal inconsistency in fake videos.
