"""
visual_features.py
Computes per-frame and aggregate visual features from face detection results.
"""

import numpy as np
import logging
from typing import List
from .face_detector import FrameFaceResult

logger = logging.getLogger(__name__)


def extract_lip_signal(face_results: List[FrameFaceResult]) -> np.ndarray:
    """
    Extract the lip movement signal (MAR per frame) as a 1D numpy array.
    Missing (no-face) frames are interpolated.
    """
    mar_values = []
    for r in face_results:
        mar_values.append(r.mar if r.detected else np.nan)

    arr = np.array(mar_values, dtype=np.float32)

    # Interpolate NaN (undetected frames)
    nans = np.isnan(arr)
    if nans.all():
        return np.zeros(len(arr), dtype=np.float32)
    if nans.any():
        indices = np.arange(len(arr))
        arr = np.interp(indices, indices[~nans], arr[~nans]).astype(np.float32)

    # Normalize to [0, 1]
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)

    return arr


def extract_visual_features(face_results: List[FrameFaceResult],
                             fps: float = 25.0) -> dict:
    """
    Compute aggregate visual features from frame-level face results.
    Returns a flat dict of scalar features.
    """
    detected = [r for r in face_results if r.detected]
    total = len(face_results)

    if not detected:
        return _empty_visual_features()

    detection_rate = len(detected) / total

    # MAR statistics
    mar_vals = np.array([r.mar for r in detected])
    # Lip velocity statistics
    vel_vals = np.array([r.lip_velocity for r in detected])
    # Head motion
    pitch_vals = np.array([r.head_pitch for r in detected])
    yaw_vals = np.array([r.head_yaw for r in detected])
    # Face area
    area_vals = np.array([r.face_area_ratio for r in detected])

    # Lip opening events (transitions from closed to open)
    lip_signal = extract_lip_signal(face_results)
    lip_events = _count_lip_events(lip_signal)

    # Lip activity ratio (fraction of frames with open mouth)
    lip_active_threshold = 0.15  # normalized MAR threshold
    lip_activity_ratio = float(np.mean(lip_signal > lip_active_threshold))

    # Temporal entropy of lip movement (complexity)
    lip_entropy = _signal_entropy(lip_signal)

    features = {
        # Detection quality
        "face_detection_rate": float(detection_rate),
        # MAR
        "mar_mean": float(np.mean(mar_vals)),
        "mar_std": float(np.std(mar_vals)),
        "mar_min": float(np.min(mar_vals)),
        "mar_max": float(np.max(mar_vals)),
        "mar_range": float(np.max(mar_vals) - np.min(mar_vals)),
        # Lip velocity
        "lip_velocity_mean": float(np.mean(vel_vals)),
        "lip_velocity_std": float(np.std(vel_vals)),
        "lip_velocity_max": float(np.max(vel_vals)),
        # Activity
        "lip_activity_ratio": lip_activity_ratio,
        "lip_event_count": float(lip_events),
        "lip_entropy": float(lip_entropy),
        # Head pose
        "head_pitch_std": float(np.std(pitch_vals)),
        "head_yaw_std": float(np.std(yaw_vals)),
        "head_motion_total": float(np.mean(np.sqrt(pitch_vals**2 + yaw_vals**2))),
        # Face area
        "face_area_mean": float(np.mean(area_vals)),
        "face_area_std": float(np.std(area_vals)),
    }
    return features


def _count_lip_events(lip_signal: np.ndarray, threshold: float = 0.2) -> int:
    """Count number of mouth-open events (zero-crossings above threshold)."""
    above = lip_signal > threshold
    count = 0
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            count += 1
    return count


def _signal_entropy(signal: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy of signal histogram — measures movement complexity. Returns [0, 1]."""
    if len(signal) == 0:
        return 0.0
    hist, _ = np.histogram(signal, bins=bins, density=False)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(bins)
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def _empty_visual_features() -> dict:
    """Return zeroed-out visual features when no face is detected."""
    return {
        "face_detection_rate": 0.0,
        "mar_mean": 0.0, "mar_std": 0.0, "mar_min": 0.0, "mar_max": 0.0, "mar_range": 0.0,
        "lip_velocity_mean": 0.0, "lip_velocity_std": 0.0, "lip_velocity_max": 0.0,
        "lip_activity_ratio": 0.0, "lip_event_count": 0.0, "lip_entropy": 0.0,
        "head_pitch_std": 0.0, "head_yaw_std": 0.0, "head_motion_total": 0.0,
        "face_area_mean": 0.0, "face_area_std": 0.0,
    }
