"""Test lag estimation on controlled shifts of a real lip signal."""
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy.signal import correlate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline.video_processor import validate_video, extract_frames, extract_audio
from backend.pipeline.face_detector import FaceDetector
from backend.pipeline.audio_features import extract_all_audio_features
from backend.pipeline.visual_features import extract_lip_signal
from backend.pipeline.sync_analyzer import resample_to_common_grid, compute_cross_correlation


def shift(x, seconds, fps):
    k = int(round(seconds * fps))
    if k == 0:
        return x.copy()
    if k > 0:
        return np.concatenate([np.zeros(k), x[:-k]])
    return np.concatenate([x[-k:], np.zeros(-k)])


def corr_with_prior(a, v, fps=25.0, max_lag=0.75, sigma=0.20):
    n = min(len(a), len(v))
    a = a[:n] - np.mean(a[:n])
    v = v[:n] - np.mean(v[:n])
    den = np.linalg.norm(a) * np.linalg.norm(v)
    c = correlate(a, v, mode="full") / den
    lags = np.arange(-(n - 1), n)
    m = np.abs(lags) <= int(max_lag * fps)
    cw, lw = c[m], lags[m]
    prior = np.exp(-0.5 * (lw / fps / sigma) ** 2)
    i = int(np.argmax(cw * prior))
    return round(float(lw[i]) / fps, 3), round(float(cw[i]), 3)


def main():
    src = sys.argv[1]
    meta = validate_video(src)
    frames, ofps, sfps = extract_frames(src, max_frames=1200, duration_hint=meta["duration"])
    det = FaceDetector()
    faces = det.process_frames(frames)
    det.close()
    work = tempfile.mkdtemp(prefix="dg_lag_")
    ap = extract_audio(src, work)
    af = extract_all_audio_features(ap) if ap else {}
    lip = extract_lip_signal(faces)
    energy = af.get("_energy_envelope", np.zeros(0))
    dur = min(meta["duration"], len(frames) / sfps if sfps else 0, len(energy) / 25.0)
    a, l = resample_to_common_grid(energy, 25.0, lip, sfps, dur, 25.0)
    print(f"n={len(a)} duration={dur:.2f}s")
    for s in [0.0, 0.04, 0.08, 0.12, 0.15, 0.2, 0.3, 0.5]:
        ls = shift(l, s, 25.0)
        no_prior = compute_cross_correlation(a, ls, 25.0)
        with_prior = corr_with_prior(a, ls)
        print(f"shift +{s*1000:.0f}ms  no_prior(lag,c)={no_prior}  prior(lag,c)={with_prior}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
