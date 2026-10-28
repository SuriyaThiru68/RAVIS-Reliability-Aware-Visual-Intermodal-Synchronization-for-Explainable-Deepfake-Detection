"""
face_detector.py
MediaPipe Face Mesh-based face and lip landmark detection.
Computes per-frame Mouth Aspect Ratio (MAR) and facial movement metrics.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh mouth landmark indices
# Upper lip: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409
# Lower lip: 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
# Mouth corners: 61, 291
# Vertical mid points: 13 (upper mid), 14 (lower mid)
MOUTH_UPPER_MID = 13
MOUTH_LOWER_MID = 14
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_UPPER_INNER = [82, 87, 312, 317]
MOUTH_LOWER_INNER = [178, 88, 402, 318]

# All landmark indices for the mouth region
MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
    270, 269, 267, 0, 37, 39, 40, 185, 13, 14, 78, 82, 312,
    87, 178, 88, 402, 318, 317, 316
]

# Nose tip for head pose reference
NOSE_TIP = 1
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362


@dataclass
class FrameFaceResult:
    detected: bool
    confidence: float = 0.0
    mar: float = 0.0                      # Mouth Aspect Ratio
    mouth_openness: float = 0.0           # Normalized vertical mouth opening
    lip_velocity: float = 0.0            # Will be filled in post-processing
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    face_area_ratio: float = 0.0         # Face area / frame area
    landmarks: Optional[np.ndarray] = None  # shape (468, 3)


class FaceDetector:
    """
    Wrapper around MediaPipe Face Mesh for per-frame face and lip analysis.
    """

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._mp = None
        self._face_mesh = None
        self._min_det = min_detection_confidence
        self._min_track = min_tracking_confidence
        self._initialized = False

    def _init_mediapipe(self):
        """Lazy initialization of MediaPipe to avoid import errors at module level."""
        try:
            import mediapipe as mp
            self._mp = mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self._min_det,
                min_tracking_confidence=self._min_track,
            )
            self._initialized = True
            logger.info("MediaPipe Face Mesh initialized")
        except ImportError:
            logger.warning("MediaPipe not available — using mock face detector")
            self._initialized = False

    def process_frames(self, frames: list) -> List[FrameFaceResult]:
        """
        Process a list of BGR frames and return a FrameFaceResult per frame.
        Computes MAR, mouth openness, and head pose for each frame.
        """
        if not self._initialized:
            self._init_mediapipe()

        if self._initialized and self._face_mesh is not None:
            return self._process_with_mediapipe(frames)
        else:
            return self._process_mock(frames)

    def _process_with_mediapipe(self, frames: list) -> List[FrameFaceResult]:
        import cv2
        results = []
        h_ref, w_ref = frames[0].shape[:2] if frames else (256, 256)

        for frame in frames:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_result = self._face_mesh.process(rgb)

            if not mp_result.multi_face_landmarks:
                results.append(FrameFaceResult(detected=False))
                continue

            face_lm = mp_result.multi_face_landmarks[0]
            lm_array = np.array(
                [[lm.x * w, lm.y * h, lm.z] for lm in face_lm.landmark],
                dtype=np.float32
            )  # shape (468, 3)

            # --- Mouth Aspect Ratio ---
            upper_mid = lm_array[MOUTH_UPPER_MID]
            lower_mid = lm_array[MOUTH_LOWER_MID]
            left_corner = lm_array[MOUTH_LEFT_CORNER]
            right_corner = lm_array[MOUTH_RIGHT_CORNER]

            vertical_dist = np.linalg.norm(upper_mid[:2] - lower_mid[:2])
            horizontal_dist = np.linalg.norm(left_corner[:2] - right_corner[:2])
            mar = vertical_dist / (horizontal_dist + 1e-6)

            # Normalized by horizontal distance
            mouth_openness = vertical_dist / (h + 1e-6)

            # --- Head Pose (simplified Euler angles from key points) ---
            nose = lm_array[NOSE_TIP]
            left_eye = lm_array[LEFT_EYE_INNER]
            right_eye = lm_array[RIGHT_EYE_INNER]

            eye_center = (left_eye + right_eye) / 2.0
            head_yaw = float(nose[0] - eye_center[0]) / (w + 1e-6)  # left-right
            head_pitch = float(nose[1] - eye_center[1]) / (h + 1e-6)  # up-down

            # --- Face area proxy (bounding box of face landmarks) ---
            lm_2d = lm_array[:, :2]
            x_min, y_min = lm_2d.min(axis=0)
            x_max, y_max = lm_2d.max(axis=0)
            face_area = (x_max - x_min) * (y_max - y_min)
            face_area_ratio = face_area / (w * h + 1e-6)

            results.append(FrameFaceResult(
                detected=True,
                confidence=0.95,
                mar=float(mar),
                mouth_openness=float(mouth_openness),
                head_pitch=float(head_pitch),
                head_yaw=float(head_yaw),
                face_area_ratio=float(face_area_ratio),
                landmarks=lm_array,
            ))

        # Compute lip velocity (frame-to-frame MAR delta)
        self._compute_lip_velocity(results)
        return results

    def _process_mock(self, frames: list) -> List[FrameFaceResult]:
        """
        Fallback mock that generates synthetic face/lip data for testing
        when MediaPipe is unavailable.
        """
        logger.warning("Using mock face detection — results are synthetic")
        n = len(frames)
        t = np.linspace(0, 2 * np.pi * 3, n)
        mar_signal = 0.25 + 0.15 * np.abs(np.sin(t)) + np.random.normal(0, 0.02, n)

        results = []
        for i, mar in enumerate(mar_signal):
            results.append(FrameFaceResult(
                detected=True,
                confidence=0.7,
                mar=float(mar),
                mouth_openness=float(mar * 0.08),
                head_pitch=float(np.random.normal(0, 0.02)),
                head_yaw=float(np.random.normal(0, 0.02)),
                face_area_ratio=0.35,
                landmarks=None,
            ))

        self._compute_lip_velocity(results)
        return results

    def _compute_lip_velocity(self, results: List[FrameFaceResult]):
        """Fill in lip_velocity for each frame (delta MAR between consecutive detected frames)."""
        prev_mar = None
        for r in results:
            if r.detected:
                if prev_mar is not None:
                    r.lip_velocity = abs(r.mar - prev_mar)
                else:
                    r.lip_velocity = 0.0
                prev_mar = r.mar
            else:
                r.lip_velocity = 0.0

    def get_detection_rate(self, results: List[FrameFaceResult]) -> float:
        """Fraction of frames where a face was successfully detected."""
        if not results:
            return 0.0
        return sum(1 for r in results if r.detected) / len(results)

    def close(self):
        if self._face_mesh is not None:
            self._face_mesh.close()
