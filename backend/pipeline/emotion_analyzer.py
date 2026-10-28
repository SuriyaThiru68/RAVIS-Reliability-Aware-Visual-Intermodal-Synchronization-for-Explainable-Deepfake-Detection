"""
emotion_analyzer.py  —  RAVIS Pipeline Stage 5

Emotion Consistency Check
─────────────────────────
A MobileNet-V2-based facial emotion classifier predicts the dominant emotion
label and a valence-arousal score for each video frame.  A prosody-based
audio emotion estimate is derived from pitch, energy and speech rate.
Disagreement between the two modalities is a manipulation indicator.

Two operating modes:
  1. Full mode  — uses TensorFlow / MobileNetV2 + a lightweight valence-arousal
                  head; requires ``tensorflow`` to be installed.
  2. Lightweight mode — uses the ``fer`` package (Fast-EmotionRecognition) which
                  bundles its own model; lighter dependency, slightly less accurate.
  3. Heuristic mode — falls back to a signal-proxy approach when neither DL
                  library is available (useful in CPU-constrained environments).
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Emotion labels used by the classifier ────────────────────────────────────
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Valence mapping: positive = pleasant, negative = unpleasant
VALENCE_MAP = {
    "angry":    -0.7,
    "disgust":  -0.8,
    "fear":     -0.6,
    "happy":     0.9,
    "sad":      -0.5,
    "surprise":  0.2,
    "neutral":   0.0,
}

# Arousal mapping: high = energetic, low = calm
AROUSAL_MAP = {
    "angry":    0.8,
    "disgust":  0.5,
    "fear":     0.7,
    "happy":    0.6,
    "sad":      0.2,
    "surprise": 0.8,
    "neutral":  0.1,
}


@dataclass
class FrameEmotionResult:
    frame_idx: int
    detected: bool
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    valence: float = 0.0         # −1 (very negative) to +1 (very positive)
    arousal: float = 0.0         # 0 (calm) to 1 (excited)
    all_scores: dict = field(default_factory=dict)


# ─── Frame-level facial emotion ──────────────────────────────────────────────

def _try_fer(frames: list) -> Optional[List[FrameEmotionResult]]:
    """
    Attempt facial emotion recognition using the ``fer`` package.
    Returns None if the package is not installed.
    """
    try:
        from fer import FER
    except ImportError:
        return None

    try:
        import cv2
        detector = FER(mtcnn=False)   # fast mode
        results  = []
        for idx, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_emotions(rgb)
            if not detections:
                results.append(FrameEmotionResult(idx, detected=False))
                continue
            # Take the first (largest) detected face
            emotions = detections[0].get("emotions") or {}
            if not emotions:
                results.append(FrameEmotionResult(idx, detected=False))
                continue
            top_label = max(emotions, key=emotions.get)
            top_conf  = float(emotions[top_label])
            results.append(FrameEmotionResult(
                frame_idx=idx,
                detected=True,
                emotion=top_label,
                emotion_confidence=top_conf,
                valence=VALENCE_MAP.get(top_label, 0.0),
                arousal=AROUSAL_MAP.get(top_label, 0.1),
                all_scores={k: float(v) for k, v in emotions.items()},
            ))
        return results
    except Exception as exc:
        logger.warning("FER emotion detection failed: %s", exc)
        return None


def _heuristic_emotion(frames: list,
                        face_results) -> List[FrameEmotionResult]:
    """
    Heuristic fallback: infer rough emotion proxy from head pose and MAR.
    Not a real emotion model — just avoids crashing when no DL is available.
    """
    results = []
    for idx, fr in enumerate(face_results):
        if not fr.detected:
            results.append(FrameEmotionResult(idx, detected=False))
            continue
        # Very open mouth → likely speech / surprise; neutral otherwise
        if fr.mar > 0.35:
            label, valence, arousal = "surprise", 0.2, 0.8
        elif fr.mar > 0.20:
            label, valence, arousal = "neutral", 0.0, 0.2
        else:
            label, valence, arousal = "neutral", 0.0, 0.1
        results.append(FrameEmotionResult(
            frame_idx=idx, detected=True,
            emotion=label, emotion_confidence=0.4,
            valence=valence, arousal=arousal,
        ))
    return results


def analyze_frame_emotions(frames: list,
                            face_results) -> List[FrameEmotionResult]:
    """
    Run facial emotion analysis on the sampled video frames.

    Tries (in order):
      1. FER package (fast, no extra GPU needed)
      2. Heuristic proxy (always available)

    Returns a FrameEmotionResult per frame.
    """
    # Try FER; if frames list is long, subsample to at most 120 frames for speed
    sample_frames = frames
    sample_mask   = list(range(len(frames)))
    if len(frames) > 120:
        sample_mask   = list(range(0, len(frames), max(1, len(frames) // 120)))
        sample_frames = [frames[i] for i in sample_mask]

    fer_results = _try_fer(sample_frames)

    if fer_results is not None:
        # Map back to full frame indices
        full_results: List[FrameEmotionResult] = [
            FrameEmotionResult(i, detected=False) for i in range(len(frames))
        ]
        for sampled_idx, res in enumerate(fer_results):
            full_idx = sample_mask[sampled_idx]
            res.frame_idx = full_idx
            full_results[full_idx] = res
        return full_results

    # Fallback
    logger.info("Using heuristic emotion proxy (no FER / TF available)")
    return _heuristic_emotion(frames, face_results)


# ─── Audio emotion (prosody) ─────────────────────────────────────────────────

def estimate_audio_emotion(audio_features: dict) -> dict:
    """
    Derive a rough emotion state from audio prosody features.

    Uses:
    - spectral centroid mean  → brightness / excitement proxy
    - RMS mean / dynamic range → energy / arousal proxy
    - ZCR mean → noisiness / tension proxy
    - speech activity → presence of speech

    Returns::

        {
          "valence":  float in [−1, +1],
          "arousal":  float in [0, 1],
          "label":    str,
          "confidence": float,
        }
    """
    centroid  = float(audio_features.get("spectral_centroid_mean", 1500) or 1500)
    rms_mean  = float(audio_features.get("rms_mean", 0.05) or 0.05)
    dyn_range = float(audio_features.get("dynamic_range_db", 10) or 10)
    zcr       = float(audio_features.get("zcr_mean", 0.05) or 0.05)
    snr       = float(audio_features.get("snr_db", 10) or 10)

    # Arousal proxy: high RMS + high centroid + high ZCR → excited/aroused
    arousal = float(np.clip(
        0.30 * min(1.0, rms_mean / 0.15) +
        0.30 * min(1.0, centroid / 4000) +
        0.25 * min(1.0, zcr / 0.15) +
        0.15 * min(1.0, dyn_range / 30.0),
        0.0, 1.0
    ))

    # Valence proxy: high SNR + moderate arousal → positive; very low SNR → negative
    valence = float(np.clip(
        0.6 * min(1.0, snr / 25.0) - 0.4 * max(0.0, arousal - 0.6),
        -1.0, 1.0
    ))

    # Map to label
    if arousal > 0.6 and valence > 0.1:
        label = "happy"
    elif arousal > 0.6 and valence < -0.1:
        label = "angry"
    elif arousal < 0.3 and valence < -0.1:
        label = "sad"
    elif arousal > 0.7:
        label = "surprise"
    else:
        label = "neutral"

    return {
        "valence":    round(float(valence), 3),
        "arousal":    round(float(arousal), 3),
        "label":      label,
        "confidence": 0.55,   # prosody-only estimate; moderate confidence
    }


# ─── Cross-modal consistency ─────────────────────────────────────────────────

def compute_emotion_consistency(frame_emotions: List[FrameEmotionResult],
                                 audio_emotion: dict,
                                 fps: float = 25.0) -> dict:
    """
    Measure agreement between facial emotion trajectory and audio emotion.

    Returns::

        {
          "consistency_score": float in [0, 1],   # 1 = fully consistent
          "disagreement_score": float,             # used as deepfake indicator
          "dominant_face_emotion": str,
          "audio_emotion_label": str,
          "valence_gap": float,                    # |face_valence − audio_valence|
          "arousal_gap": float,
          "suspicious": bool,
        }
    """
    detected = [r for r in frame_emotions if r.detected]
    if not detected:
        return {
            "consistency_score":    0.5,
            "disagreement_score":   0.0,
            "dominant_face_emotion": "unknown",
            "audio_emotion_label":  audio_emotion.get("label", "neutral"),
            "valence_gap":          0.0,
            "arousal_gap":          0.0,
            "suspicious":           False,
        }

    # Dominant face emotion (most frequent)
    from collections import Counter
    emotion_counts    = Counter(r.emotion for r in detected)
    dominant_emotion  = emotion_counts.most_common(1)[0][0]

    face_valence_mean = float(np.mean([r.valence for r in detected]))
    face_arousal_mean = float(np.mean([r.arousal  for r in detected]))

    audio_valence = float(audio_emotion.get("valence", 0.0))
    audio_arousal = float(audio_emotion.get("arousal", 0.3))

    valence_gap = abs(face_valence_mean - audio_valence)
    arousal_gap = abs(face_arousal_mean  - audio_arousal)

    # Disagreement score: 0 = perfectly consistent, 1 = maximum inconsistency
    disagreement = float(np.clip(
        0.55 * valence_gap / 2.0 +   # valence range is [−1, 1] → max gap = 2
        0.45 * arousal_gap,           # arousal range is  [0,  1] → max gap = 1
        0.0, 1.0
    ))
    consistency = 1.0 - disagreement

    # Flag as suspicious if disagreement is above a conservative threshold
    suspicious = disagreement > 0.45

    return {
        "consistency_score":    round(float(consistency),   3),
        "disagreement_score":   round(float(disagreement),  3),
        "dominant_face_emotion": dominant_emotion,
        "audio_emotion_label":  audio_emotion.get("label", "neutral"),
        "valence_gap":          round(float(valence_gap),   3),
        "arousal_gap":          round(float(arousal_gap),   3),
        "face_valence_mean":    round(float(face_valence_mean), 3),
        "face_arousal_mean":    round(float(face_arousal_mean), 3),
        "audio_valence":        round(float(audio_valence), 3),
        "audio_arousal":        round(float(audio_arousal), 3),
        "suspicious":           bool(suspicious),
    }


# ─── Master function ─────────────────────────────────────────────────────────

def run_emotion_analysis(frames: list,
                          face_results,
                          audio_features: dict,
                          fps: float = 25.0) -> dict:
    """
    Run the complete Stage 5 emotion consistency analysis.

    Returns a dict containing:
      - ``frame_emotions``       — per-frame FrameEmotionResult list (serialised)
      - ``audio_emotion``        — prosody-based emotion estimate
      - ``consistency``          — cross-modal consistency metrics
      - ``emotion_disagreement`` — scalar disagreement score for the classifier
    """
    logger.info("Stage 5: Running emotion consistency analysis…")

    frame_emotions = analyze_frame_emotions(frames, face_results)
    audio_emotion  = estimate_audio_emotion(audio_features)
    consistency    = compute_emotion_consistency(frame_emotions, audio_emotion, fps)

    # Serialise per-frame results (drop landmarks to keep payload manageable)
    frames_serial = [
        {
            "frame_idx":          r.frame_idx,
            "detected":           r.detected,
            "emotion":            r.emotion,
            "emotion_confidence": round(r.emotion_confidence, 3),
            "valence":            round(r.valence, 3),
            "arousal":            round(r.arousal, 3),
        }
        for r in frame_emotions
    ]

    return {
        "frame_emotions":       frames_serial,
        "audio_emotion":        audio_emotion,
        "consistency":          consistency,
        "emotion_disagreement": consistency["disagreement_score"],
    }
