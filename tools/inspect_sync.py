"""Inspect the audio/lip correlation curve for a video (debug helper)."""
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy.signal import correlate, find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline.video_processor import validate_video, extract_frames, extract_audio
from backend.pipeline.face_detector import FaceDetector
from backend.pipeline.audio_features import extract_all_audio_features
from backend.pipeline.visual_features import extract_lip_signal
from backend.pipeline.sync_analyzer import resample_to_common_grid, compute_cross_correlation


def main():
    src = sys.argv[1]
    meta = validate_video(src)
    frames, ofps, sfps = extract_frames(src, max_frames=1200, duration_hint=meta["duration"])
    det = FaceDetector()
    faces = det.process_frames(frames)
    det.close()
    work = tempfile.mkdtemp(prefix="dg_sync_")
    ap = extract_audio(src, work)
    af = extract_all_audio_features(ap) if ap else {}
    lip = extract_lip_signal(faces)
    energy = af.get("_energy_envelope", np.zeros(0))
    adur = af.get("_audio_duration", 0.0)
    dur = min(meta["duration"], len(frames) / sfps if sfps else 0)
    if len(energy):
        dur = min(dur, len(energy) / 25.0)
    a, l = resample_to_common_grid(energy, 25.0, lip, sfps, dur, 25.0)
    n = len(a)
    print(f"duration={dur:.2f}s n={n} sfps={sfps:.2f} ofps={ofps:.2f} "
          f"lip_std={np.std(l):.4f} energy_std={np.std(a):.4f} "
          f"lip_range={np.ptp(l):.4f} frame_face_rate={np.mean([f.detected for f in faces]):.2f}")

    # correlation curve
    aa = a - a.mean()
    ll = l - l.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(ll)
    if denom < 1e-8:
        print("flat signal")
        return
    c = correlate(aa, ll, mode='full') / denom
    lags = np.arange(-(n - 1), n)
    print("corr at ms:", {int(ms): round(float(c[np.argmin(np.abs(lags - ms / 1000 * 25))]), 4)
                          for ms in [0, 40, 80, 120, 200, 300, 400, 500, 650, 750]})
    sel = np.abs(lags) <= int(0.75 * 25)
    cs, ls = c[sel], lags[sel]
    peaks, _ = find_peaks(cs)
    order = np.argsort(cs[peaks])[::-1][:8]
    print("top peaks:", [(round(float(ls[peaks[i]]) / 25 * 1000), round(float(cs[peaks[i]]), 3)) for i in order])
    print("xcorr(restricted):", compute_cross_correlation(a, l, 25.0))
    print("xcorr(max 0.15s):", compute_cross_correlation(a, l, 25.0, 0.15))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
