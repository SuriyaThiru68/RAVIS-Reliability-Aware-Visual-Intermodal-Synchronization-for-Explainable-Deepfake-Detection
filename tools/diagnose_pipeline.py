"""
Temporary diagnostic: run the detection stages on a video and compare the
legacy (buggy) sync features against a correctly-resampled 25 Hz timebase.

Usage:  python tools/diagnose_pipeline.py <video> [max_frames]
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline.video_processor import validate_video, extract_frames, extract_audio, TARGET_FPS
from backend.pipeline.face_detector import FaceDetector
from backend.pipeline.audio_features import extract_all_audio_features
from backend.pipeline.visual_features import extract_visual_features, extract_lip_signal
from backend.pipeline.sync_analyzer import (
    compute_sync_features, compute_cross_correlation, compute_per_second_sync,
)
from backend.pipeline.classifier import classify, build_feature_vector, load_model


def resample_common(audio_energy, lip_signal, audio_fps, lip_fps, duration, common_fps=25.0):
    n = max(2, int(round(duration * common_fps)))
    t = np.arange(n) / common_fps
    a_t = np.arange(len(audio_energy)) / audio_fps
    l_t = np.arange(len(lip_signal)) / lip_fps
    a = np.interp(t, a_t, audio_energy, left=0.0, right=0.0)
    l = np.interp(t, l_t, lip_signal, left=lip_signal[0] if len(lip_signal) else 0.0,
                  right=lip_signal[-1] if len(lip_signal) else 0.0)
    return a.astype(np.float64), l.astype(np.float64)


def main():
    path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 750

    t0 = time.time()
    meta = validate_video(path)
    print("META:", meta)
    if not meta["valid"]:
        return

    frames, original_fps = extract_frames(path, max_frames=max_frames)
    interval = max(1, int(round(original_fps / TARGET_FPS)))
    sampling_fps = original_fps / interval
    print(f"frames={len(frames)} original_fps={original_fps:.3f} interval={interval} "
          f"sampling_fps={sampling_fps:.3f} covered={len(frames)/sampling_fps:.2f}s "
          f"duration={meta['duration']}s ({time.time()-t0:.1f}s)")

    detector = FaceDetector()
    face_results = detector.process_frames(frames)
    detector.close()
    print(f"face detection done ({time.time()-t0:.1f}s)")

    work = tempfile.mkdtemp(prefix="dg_diag_")
    audio_path = extract_audio(path, work)
    afeats = extract_all_audio_features(audio_path) if audio_path else {}
    vfeats = extract_visual_features(face_results, fps=sampling_fps)
    lip = extract_lip_signal(face_results)
    energy = afeats.get("_energy_envelope", np.zeros(0))
    audio_dur = afeats.get("_audio_duration", 0.0)
    print(f"lip_len={len(lip)} lip_fps={sampling_fps:.3f} lip_dur={len(lip)/sampling_fps:.2f}s "
          f"energy_len={len(energy)} energy_fps=25 energy_dur={audio_dur:.2f}s")

    # ── Legacy (current) path ─────────────────────────────────────────────
    legacy = compute_sync_features(
        audio_energy=energy, lip_signal=lip,
        audio_events=afeats.get("_audio_events", []), fps=25.0,
    )
    base = {}
    base.update({k: v for k, v in afeats.items() if not k.startswith("_")})
    base.update(vfeats)
    base.update({k: v for k, v in legacy.items() if not k.startswith("_")})
    old_cls = classify(base)
    print("\nLEGACY:")
    print({k: round(float(v), 4) for k, v in legacy.items() if not k.startswith("_")})
    print(f"legacy per-second windows={len(legacy['_per_second_sync'])} "
          f"suspicious={len(legacy['_suspicious_timestamps'])}")
    print("LEGACY CLS:", {k: old_cls[k] for k in ("verdict", "deepfake_probability", "confidence")})

    # ── Fixed path: common 25 Hz grid, whole covered duration ─────────────
    duration = min(len(frames) / sampling_fps, audio_dur if audio_dur > 0 else 1e9)
    a25, l25 = resample_common(energy, lip, 25.0, sampling_fps, duration, 25.0)
    fixed = compute_sync_features(
        audio_energy=a25, lip_signal=l25,
        audio_events=afeats.get("_audio_events", []), fps=25.0,
    )
    base2 = {}
    base2.update({k: v for k, v in afeats.items() if not k.startswith("_")})
    base2.update(vfeats)
    base2.update({k: v for k, v in fixed.items() if not k.startswith("_")})
    new_cls = classify(base2)
    print("\nFIXED (25 Hz common grid):")
    print({k: round(float(v), 4) for k, v in fixed.items() if not k.startswith("_")})
    print(f"fixed per-second windows={len(fixed['_per_second_sync'])} "
          f"suspicious={len(fixed['_suspicious_timestamps'])}")
    print("FIXED CLS:", {k: new_cls[k] for k in ("verdict", "deepfake_probability", "confidence")})

    print("\nTOP FEATURE IMPORTANCES:", [f"{k}:{v:.3f}" for k, v in new_cls["top_features"][:6]])
    print(f"total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
