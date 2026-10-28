# RAVIS: Reliability-Aware Visual-Intermodal Synchronization for Explainable Deepfake Detection

## 1. Product Vision

RAVIS is a forensic-style explainable deepfake detection dashboard that helps users understand not just whether a video is fake, but why it is fake. The interface combines video playback, facial landmark analysis, audio signal analysis, temporal synchronization graphs, and reliability-aware evidence to support forensic interpretation.

The design goal is to make complex multimodal evidence understandable to both technical and non-technical users.

---

## 2. Core Design Principles

- Explainability first: every prediction is supported by visual and signal evidence.
- Multimodal clarity: video, audio, and synchronization are always visible together.
- Temporal awareness: suspicious events are shown along a time axis.
- Trust calibration: confidence and reliability indicators are presented clearly and honestly.
- Forensic presentation: the UI should feel like a professional investigative dashboard.

---

## 3. Overall Layout

### 3.1 Page Structure
                    
The application should be a single-page dashboard with a top navigation bar and four major sections:

1. Header + upload controls
2. Prediction summary panel
3. Media analysis section
4. Evidence and synchronization analysis section
5. Temporal timeline and explainability section

### 3.2 Layout Composition

- Header: 72px height
- Main workspace: large panel-based dashboard
- Left column: 38% width
- Right column: 62% width
- Bottom evidence panels: full width with grid layout

---

## 4. Header / Toolbar

### Header content

- Left: logo and app name: RAVIS
- Center: file metadata or “Analysis Session” state
- Right: upload button, save report, export report, refresh analysis

### Upload option requirements

The upload panel should be prominent and visually accessible.

#### Primary upload controls
- Drag-and-drop upload area
- Click-to-browse file selector
- Supported formats: MP4, AVI, MOV, MKV, WEBM
- Display file name, duration, size, and resolution
- Show a “Analyze” button only after a valid file is selected

#### Upload panel visual style
- Large dashed drop zone
- Blue accent gradient for primary action
- Subtle hover and focus states
- File validation message below the upload box

Example:
- “Upload video sample”
- “Supported: MP4 / AVI / MOV / MKV / WEBM”
- “Max size: 500MB”

---

## 5. Prediction Summary Panel

This is the top verdict area and should appear immediately after upload.

### Summary card contents

- Verdict label: REAL / FAKE
- Confidence score: e.g. 96.4%
- Reliability score: e.g. 0.89
- Final category: Authentic / Deepfake / Uncertain
- Timestamp summary: suspicious segments
- Quick reason summary: “Temporal desynchronization detected”

### Visual design

- Left side: large verdict status badge
- Right side: probability bars and reliability score
- Background color based on result:
  - Real: green / teal
  - Fake: red / crimson
  - Uncertain: amber / yellow

### Example verdict panel

- Verdict: FAKE
- Confidence: 94.7%
- Reliability: High confidence
- Evidence count: 6 suspicious indicators found

---

## 6. Media Analysis Section

This section includes the main visual investigation area. It should be split into two halves:

- Left: video playback and face/lip analysis
- Right: audio spectrogram and frequency analysis

## 6.1 Video Playback and Face / Lip Mesh Panel

### Layout

Large video player occupying the top-left major panel, with overlay controls:
- play/pause
- timeline scrubber
- current timestamp
- resolution metadata

### Overlay cues

- MediaPipe face mesh overlay on the face region
- Lip landmarks highlighted in bright color
- Frame markers to indicate suspicious timestamps
- Time cursor synced with graph below

### Visual requirements

- Show the video with a transparent face mesh overlay
- Highlight mouth region in a distinct cyan or orange color
- Add a frame selection marker when the user clicks suspicious timestamps
- Use a timeline ruler under the video to indicate detected anomalies

### Deepfake evidence display

Under the video, show a small evidence strip with suspicious timestamps, such as:
- 00:02:14 – mouth motion mismatch
- 00:02:46 – audio lag detected
- 00:03:31 – face landmark drift

### “Why it is fake” chips

- Lip sync mismatch
- Audio-video lag
- Landmark jitter
- Unexpected mouth motion pattern

---

## 6.2 Audio Spectrogram and Frequency Energy Panel

Positioned beside the video panel.

### Main features

- Spectrogram heatmap across time and frequency
- Frequency energy curve underneath
- Highlight suspicious frequency bands
- Mark segments where audio has been replaced or dubbed

### Audio forensics indicators

- Strong spectral artifacts
- Frequency inconsistencies over time
- Energy drop or sudden waveform changes
- Voice mismatch or synthetic speech artifacts

### Visual style

- Dark background for signal display
- Bright colors for meaningful activity
- Green and red overlays for synchronized vs desynchronized regions
- A cursor line to align with the selected video frame

---

## 7. Explainability and Forensic Evidence Panel

This is the second row of the main dashboard and should compare the prediction evidence by modality.

### 7.1 Real / Fake Evidence Blocks

A grid of cards like:

- Audio track forensics
- Face and pixel forensics
- Lip sync kinematics
- Temporal consistency
- Motion irregularity
- Compression artifact analysis

Each card contains:
- title
- short explanation
- confidence indicator
- one visual mini-chart or bar

### Visual design

- Card background: neutral dark or light depending on theme
- Strong label and confidence meter
- Accent colors for attack type classification

### Card examples

#### Audio track forensics
- Synthetic speech artifacts detected
- Frequency instability over time
- Spectral mismatch detected

#### Face and pixel forensics
- Facial landmark drift
- Eye and mouth region distortion
- Inconsistent texture or edge artifacts

#### Lip sync kinematics
- Mouth opening does not match phoneme timing
- Beat mismatch between audio envelope and face motion
- Temporal lag detected near word boundaries

---

## 8. Audio-Visual Synchronization Graph

This is one of the core components and should be visually dominant.

### Requirement 1: Green Zone for Authentic Human Co-Articulation

When the audio and lip motion are synchronized, the graph should display a stable, smooth, well-aligned signal in the green region.

#### Visual representation
- X-axis: time
- Y-axis: synchronization index / amplitude
- Green region: authentic co-articulation zone
- The curve should stay close to the center or within green band

#### Design cues
- Green fill area for authentic human alignment
- Smoother waveform with natural phase transitions
- Lower variance around alignment peaks

### Label example

“GREEN ZONE: AUTHENTIC HUMAN CO-ARTICULATION”

---

## 9. Temporal Alignment Timeline

This should sit directly below or adjacent to the synchronization graph.

### Requirement 2 and 8

The timeline should be a horizontal time strip with color-coded segments:
- Green = synchronized speech
- Red = desynchronized speech or dubbing mismatch

#### Design requirements
- Time axis spans the full video duration
- Each segment is mapped to the specific speech or mouth motion region
- User can hover over segments and see details
- A vertical scrubber line indicates current selected time

### Interpretation

- Green intervals = audio and lip movement align naturally
- Red intervals = inconsistent phoneme timing or dubbing mismatch
- User sees exact where the manipulation likely occurs

Example labels:
- GREEN = SYNCHRONOUS
- RED = DESYNCHRONIZED SPEECH

---

## 10. Red-Zone Desynchronization View

### Requirement 7

This is the negative case representation and should be shown as an alternate diagnostic mode or as a highlighted segment in the same graph.

### Visual features

- Curve drifts outside the green band
- Large phase lag between audio envelope and mouth motion
- Red region highlights mismatch windows
- Red waveform spikes or disconnects from face motion

### Label example

“AUDIO-VISUAL SYNCHRONIZATION GRAPH (RED ZONE: DESYNCHRONIZATION / DUBBING DETECTED)”

### What the user sees

- Sudden temporal offset
- Mouth opens while audio energy is absent
- Overlapping mismatched segments
- Speech alignment failure at certain time windows

---

## 11. Detailed Interface Breakdown

## 11.1 Top-left panel: Upload + file inspector

Contents:
- file input area
- selected path
- video duration
- frames per second
- audio length
- prediction status

## 11.2 Top-center panel: Verdict summary

Contents:
- overall fake probability
- reliability score
- explanation summary

## 11.3 Middle-left panel: Video with face mesh

Contents:
- video playback
- face mesh overlay
- mouth landmarks
- suspicious frame markers

## 11.4 Middle-right panel: Audio signal analysis

Contents:
- spectrogram
- waveform
- frequency energy pattern
- suspicious audio segments

## 11.5 Bottom-left panel: Synchronization graph

Contents:
- green authentic region
- red desync region
- temporal correlation curve

## 11.6 Bottom-right panel: Timeline + evidence log

Contents:
- synchronized vs unsynchronized segments
- suspicious window annotations
- predicted fake region list

---

## 12. Color System

### Primary palette
- Deep navy: #081A2C
- Teal: #12C9C6
- Cyan: #3EC8FF
- Green: #26D07C
- Red: #FF5E5B
- Amber: #FFB020
- Light gray: #EEF3F7
- Dark panel: #111F2D

### Status colors
- Real / authentic: green
- Fake / manipulated: red
- Uncertain: amber
- Neutral background: slate navy

---

## 13. Interaction Patterns

### Hover states
- Hover on the timeline to inspect exact frame timing
- Hover on graph line to see audio-lip lag value
- Hover on evidence cards to show deeper explanation

### Click behavior
- Click graph segment to jump to exact video frame
- Click suspicious timeline region to inspect local evidence
- Click audio segment to compare with lip motion window

### User flow
1. Upload video
2. View prediction verdict
3. Inspect video and face mesh
4. Review audio spectrogram
5. Study synchronization graph
6. Read timeline mismatches
7. Review explainability cards
8. Export final explanation report

---

## 14. Suggested Final Dashboard Composition

### Row 1
- Upload / file status panel
- Verdict summary panel

### Row 2
- Video playback and lip mesh panel
- Audio spectrogram / frequency energy panel

### Row 3
- Explainability cards grid
- Real vs fake evidence analysis

### Row 4
- Synchronization graph panel
- Temporal alignment timeline panel

---

## 15. Final UX Statement

The RAVIS UI should feel like a forensic investigation cockpit. It is not a simple prediction dashboard; it is a visual reasoning tool that shows how deepfake evidence emerges across video, audio, and temporal synchronization. The interface should make truth discovery intuitive by translating technical signal mismatches into clear, explainable visual evidence.

---

## 16. Recommended Visual Design Summary

- Dark forensic dashboard theme
- High-contrast charts and overlays
- Clear green/red synchronization states
- Time-synchronized media player and signal graphs
- Strong evidence cards for explainability
- Clean, professional, academic UI with analytics feel

---

## 17. Example Final Look and Feel

The application should look like:
- A high-end analytics dashboard
- A digital forensics system
- An explainable multimodal AI interface
- A high-precision evidence review tool

The design should prioritize trust, evidence, time alignment, and user clarity above all else.






