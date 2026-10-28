"""
sync_analyzer.py
Temporal alignment and audio-visual synchronization analysis.
Core module for detecting deepfake manipulation via A/V sync mismatch.
"""

import numpy as np
import logging
from scipy.signal import correlate
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

# Window sizes for local sync analysis
WINDOW_SIZE_SECONDS = 1.0   # 1-second sliding window
STRIDE_SECONDS = 0.5        # 50% overlap
MIN_WINDOW_FRAMES = 10

# Human speech cannot be offset by more than ~0.75 s and still be considered
# "the same utterance".  Searching the full correlation range makes a random
# weak peak 10+ seconds away look like a real lag, which used to poison the
# classifier for every clip.
MAX_LAG_SECONDS = 0.75


def resample_to_common_grid(audio_energy: np.ndarray,
                            audio_fps: float,
                            lip_signal: np.ndarray,
                            lip_fps: float,
                            duration: float,
                            common_fps: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Put the audio envelope and the lip signal on one shared time axis.

    ``compute_energy_envelope`` emits samples at 25 Hz over the audio duration,
    while the lip signal is emitted at the *frame sampling rate* (which can be
    12, 15, 24, 30...).  Comparing those arrays directly multiplies every lag by
    (25 / lip_fps) and makes the per-second timeline cover only a fraction of
    the clip.  This helper resamples both signals onto a common grid that spans
    exactly ``duration`` seconds.
    """
    audio_energy = np.asarray(audio_energy, dtype=np.float64).ravel()
    lip_signal = np.asarray(lip_signal, dtype=np.float64).ravel()
    duration = float(duration or 0.0)
    if duration <= 0 or audio_energy.size == 0 or lip_signal.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    n = max(2, int(round(duration * common_fps)))
    t_common = np.arange(n, dtype=np.float64) / common_fps

    a_times = np.arange(audio_energy.size, dtype=np.float64) / max(float(audio_fps), 1e-6)
    l_times = np.arange(lip_signal.size, dtype=np.float64) / max(float(lip_fps), 1e-6)

    a = np.interp(t_common, a_times, audio_energy, left=0.0, right=0.0)
    l = np.interp(t_common, l_times, lip_signal,
                  left=float(lip_signal[0]), right=float(lip_signal[-1]))
    return a, l


def compute_cross_correlation(audio_energy: np.ndarray,
                               lip_signal: np.ndarray,
                               sample_rate: float = 25.0,
                               max_lag_seconds: float = MAX_LAG_SECONDS) -> tuple[float, float]:
    """
    Compute normalized cross-correlation between audio energy and lip signal.

    The search is restricted to ``+/-max_lag_seconds`` so a weak random peak far
    away cannot masquerade as a physical offset.  Returns
    ``(peak_correlation, lag_seconds)`` with a signed lag (positive = lip
    movement lags the audio).
    """
    if audio_energy is None or lip_signal is None:
        return 0.0, 0.0

    audio_energy = np.asarray(audio_energy, dtype=np.float64)
    lip_signal = np.asarray(lip_signal, dtype=np.float64)
    if audio_energy.size == 0 or lip_signal.size == 0:
        return 0.0, 0.0

    # Align lengths
    min_len = min(len(audio_energy), len(lip_signal))
    if min_len < 2:
        return 0.0, 0.0

    a = audio_energy[:min_len].copy()
    v = lip_signal[:min_len].copy()

    # Zero-mean / unit-variance so silence or a flat mouth signal cannot
    # produce a spurious normalisation.
    a = a - np.mean(a)
    v = v - np.mean(v)
    norm_a = np.linalg.norm(a)
    norm_v = np.linalg.norm(v)
    if norm_a < 1e-8 or norm_v < 1e-8:
        return 0.0, 0.0

    corr = correlate(a, v, mode='full')
    corr = corr / (norm_a * norm_v)
    lags = np.arange(-(min_len - 1), min_len)
    max_lag_frames = max(1, int(round(max_lag_seconds * max(sample_rate, 1e-6))))
    window = np.abs(lags) <= max_lag_frames
    if not window.any():
        window = np.ones_like(lags, dtype=bool)

    corr_win = corr[window]
    lags_win = lags[window]

    # 1) Sync *strength* is measured close to zero lag.  Searching far away
    #    and taking the maximum lets periodic mouth motion (or the zero padding
    #    from a shifted signal) create a large value that has nothing to do
    #    with actual synchronisation.
    near_frames = max(1, int(round(0.12 * max(sample_rate, 1e-6))))
    near_mask = np.abs(lags_win) <= near_frames
    if not near_mask.any():
        near_mask = lags_win == 0
    near_corr = corr_win[near_mask]
    near_val = float(np.max(near_corr))
    xcorr_peak = float(np.clip(near_val, 0.0, 1.0))

    # 2) Offset is the strongest peak over the wider search window.  Only
    #    trust it if it rises above noise; otherwise report no offset.
    best_idx = int(np.argmax(corr_win))
    best_val = float(corr_win[best_idx])
    if best_val < 0.12:
        lag_seconds = 0.0
    else:
        lag_frames = float(lags_win[best_idx])
        if 0 < best_idx < len(corr_win) - 1:
            y0 = float(corr_win[best_idx - 1])
            y1 = best_val
            y2 = float(corr_win[best_idx + 1])
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom
                if abs(delta) <= 1.0:
                    lag_frames += delta
        lag_seconds = lag_frames / max(sample_rate, 1e-6)

    return xcorr_peak, float(lag_seconds)


def compute_dtw_distance(audio_energy: np.ndarray,
                          lip_signal: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between audio and lip signals.

    Tries FastDTW (O(n) approximate) first; falls back to the exact O(nm)
    implementation if the ``fastdtw`` package is not installed.
    Lower distance = better alignment.
    """
    if audio_energy is None or lip_signal is None:
        return 0.0

    audio_energy = np.asarray(audio_energy, dtype=np.float64)
    lip_signal = np.asarray(lip_signal, dtype=np.float64)
    if audio_energy.size == 0 or lip_signal.size == 0:
        return 0.0

    a = audio_energy.copy()
    v = lip_signal.copy()

    # z-normalise so amplitude differences don't dominate
    if np.std(a) > 1e-8:
        a = (a - np.mean(a)) / np.std(a)
    if np.std(v) > 1e-8:
        v = (v - np.mean(v)) / np.std(v)

    n, m = len(a), len(v)
    if n == 0 or m == 0:
        return 0.0

    # ?? Try FastDTW ??????????????????????????????????????????????????????????
    try:
        from fastdtw import fastdtw
        dist, _ = fastdtw(a, v, dist=lambda x, y: abs(x - y))
        return float(dist) / (n + m)
    except ImportError:
        pass  # fall through to exact DTW below

    # ?? Exact DTW (O(nm)) -- downsampled to 300 pts for speed ????????????????
    max_len = 300
    if len(a) > max_len:
        idx = np.linspace(0, len(a) - 1, max_len).astype(int)
        a = a[idx]
    if len(v) > max_len:
        idx = np.linspace(0, len(v) - 1, max_len).astype(int)
        v = v[idx]

    n, m = len(a), len(v)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - v[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    dist = float(dtw_matrix[n, m])
    return dist / (n + m)


def compute_cosine_similarity_windowed(audio_energy: np.ndarray,
                                        lip_signal: np.ndarray,
                                        fps: float = 25.0) -> tuple[float, np.ndarray]:
    """
    Compute average cosine similarity over sliding windows.
    Returns (mean_similarity, per_window_similarities)
    """
    if audio_energy is None or lip_signal is None:
        return 0.5, np.array([0.5], dtype=np.float32)

    audio_energy = np.asarray(audio_energy, dtype=np.float64)
    lip_signal = np.asarray(lip_signal, dtype=np.float64)
    if audio_energy.size == 0 or lip_signal.size == 0:
        return 0.5, np.array([0.5], dtype=np.float32)

    min_len = min(len(audio_energy), len(lip_signal))
    if min_len == 0:
        return 0.5, np.array([0.5], dtype=np.float32)

    window_frames = max(MIN_WINDOW_FRAMES, int(WINDOW_SIZE_SECONDS * fps))
    stride_frames = max(1, int(STRIDE_SECONDS * fps))

    similarities = []
    for start in range(0, min_len - window_frames + 1, stride_frames):
        end = start + window_frames
        a_win = audio_energy[start:end].astype(np.float64)
        v_win = lip_signal[start:end].astype(np.float64)

        if np.std(a_win) < 1e-6 or np.std(v_win) < 1e-6:
            similarities.append(0.5)
            continue

        a_win = (a_win - np.mean(a_win)) / (np.std(a_win) + 1e-8)
        v_win = (v_win - np.mean(v_win)) / (np.std(v_win) + 1e-8)

        try:
            sim = 1.0 - cosine(a_win, v_win)
        except ValueError:
            sim = 0.5
        sim = max(0.0, min(1.0, sim))
        similarities.append(float(sim))

    if not similarities:
        return 0.5, np.array([0.5], dtype=np.float32)

    return float(np.mean(similarities)), np.array(similarities, dtype=np.float32)


def compute_per_second_sync(audio_energy: np.ndarray,
                             lip_signal: np.ndarray,
                             fps: float = 25.0) -> np.ndarray:
    """
    Compute sync score for each 1-second window.
    Returns array of per-second scores in [0, 1] (1 = in sync, 0 = out of sync).
    """
    if audio_energy is None or lip_signal is None:
        return np.array([], dtype=np.float32)

    audio_energy = np.asarray(audio_energy, dtype=np.float64)
    lip_signal = np.asarray(lip_signal, dtype=np.float64)
    if audio_energy.size == 0 or lip_signal.size == 0:
        return np.array([], dtype=np.float32)

    min_len = min(len(audio_energy), len(lip_signal))
    if min_len == 0:
        return np.array([], dtype=np.float32)

    frames_per_sec = int(fps)
    if frames_per_sec <= 0:
        return np.array([], dtype=np.float32)

    n_seconds = min_len // frames_per_sec
    scores = []
    for sec in range(n_seconds):
        start = sec * frames_per_sec
        end = start + frames_per_sec
        a_win = audio_energy[start:end].astype(np.float64)
        v_win = lip_signal[start:end].astype(np.float64)

        if np.std(a_win) < 1e-6 or np.std(v_win) < 1e-6:
            scores.append(0.6)
            continue

        a_norm = (a_win - np.mean(a_win)) / (np.std(a_win) + 1e-8)
        v_norm = (v_win - np.mean(v_win)) / (np.std(v_win) + 1e-8)

        try:
            corr = np.corrcoef(a_norm, v_norm)[0, 1]
        except ValueError:
            corr = 0.0
        score = float(max(0.0, (corr + 1.0) / 2.0))
        scores.append(score)

    return np.array(scores, dtype=np.float32)


def compute_sync_features(audio_energy: np.ndarray,
                           lip_signal: np.ndarray,
                           audio_events: list,
                           fps: float = 25.0) -> dict:
    """
    Master sync feature computation.
    Returns dict with all synchronization features.
    """
    if audio_energy is None or lip_signal is None:
        audio_energy = np.array([], dtype=np.float64)
        lip_signal = np.array([], dtype=np.float64)

    audio_energy = np.asarray(audio_energy, dtype=np.float64)
    lip_signal = np.asarray(lip_signal, dtype=np.float64)
    audio_events = list(audio_events or [])

    if audio_energy.size == 0 or lip_signal.size == 0:
        empty = np.array([], dtype=np.float32)
        return {
            "xcorr_peak": 0.0,
            "audio_lip_lag_seconds": 0.0,
            "audio_lip_lag_abs": 0.0,
            "dtw_distance": 0.0,
            "dtw_score": 0.5,
            "cosine_similarity_mean": 0.5,
            "cosine_similarity_std": 0.0,
            "cosine_similarity_min": 0.5,
            "sync_score": 0.5,
            "alignment_score": 0.5,
            "suspicious_window_fraction": 0.0,
            "_per_second_sync": empty,
            "_suspicious_timestamps": [],
            "_audio_events": audio_events,
            "_sync_score_raw": 0.5,
            "_active_second_count": 0,
        }

    # Align signals
    min_len = min(len(audio_energy), len(lip_signal))
    audio_energy = audio_energy[:min_len]
    lip_signal = lip_signal[:min_len]

    xcorr_peak, lag_seconds = compute_cross_correlation(
        audio_energy, lip_signal, sample_rate=fps
    )

    # A constant container-level offset is not frame-by-frame tampering.
    # Align it out before measuring local coherence; the lag itself is still
    # reported as a feature.
    lag_frames = int(round(lag_seconds * fps))
    if 0 < lag_frames < min_len - 1:
        a_aligned = audio_energy[lag_frames:]
        l_aligned = lip_signal[:min_len - lag_frames]
    elif -min_len + 1 < lag_frames < 0:
        k = -lag_frames
        a_aligned = audio_energy[:min_len - k]
        l_aligned = lip_signal[k:]
    else:
        a_aligned = audio_energy
        l_aligned = lip_signal

    dtw_dist = compute_dtw_distance(a_aligned, l_aligned)
    cos_sim_mean, cos_sim_per_window = compute_cosine_similarity_windowed(a_aligned, l_aligned, fps)
    per_second_sync = compute_per_second_sync(a_aligned, l_aligned, fps)

    # Which seconds actually contain speech?  Silent windows cannot provide
    # sync evidence and must not be counted as suspicious.
    frames_per_sec = max(1, int(round(fps)))
    n_seconds = len(per_second_sync)
    active_mask = np.zeros(n_seconds, dtype=bool)
    if n_seconds and len(a_aligned) >= frames_per_sec:
        usable = min(len(a_aligned), n_seconds * frames_per_sec)
        sec_energy = a_aligned[:usable].reshape(n_seconds, frames_per_sec).mean(axis=1)
        peak_energy = float(np.max(sec_energy)) if sec_energy.size else 0.0
        if peak_energy > 1e-4:
            active_mask = sec_energy > max(1e-4, 0.12 * peak_energy)
    active_seconds = int(active_mask.sum())

    sync_threshold = 0.45
    if active_seconds:
        alignment_score = float(np.mean(per_second_sync[active_mask] > sync_threshold))
        suspicious_seconds = [
            i for i in range(n_seconds)
            if active_mask[i] and per_second_sync[i] < 0.35
        ]
        suspicious_fraction = len(suspicious_seconds) / active_seconds
    else:
        alignment_score = 0.5   # neutral: nothing to synchronise
        suspicious_seconds = []
        suspicious_fraction = 0.0

    dtw_score = float(np.exp(-dtw_dist * 2))
    if active_seconds == 0:
        # No detected speech: never report "maximum suspicion" by default.
        dtw_score = 0.5
        sync_score = 0.5
    else:
        sync_score = float(0.40 * xcorr_peak + 0.35 * cos_sim_mean + 0.25 * dtw_score)
    sync_score = max(0.0, min(1.0, sync_score))

    suspicious_threshold = 0.35
    suspicious_timestamps = []
    for i, score in enumerate(per_second_sync):
        if active_mask[i] and score < suspicious_threshold:
            suspicious_timestamps.append({
                "second": i,
                "timestamp": f"{i//60:02d}:{i%60:02d}",
                "sync_score": round(float(score), 3),
            })

    features = {
        "xcorr_peak": float(xcorr_peak),
        "audio_lip_lag_seconds": float(lag_seconds),
        "audio_lip_lag_abs": float(abs(lag_seconds)),
        "dtw_distance": float(dtw_dist),
        "dtw_score": float(dtw_score),
        "cosine_similarity_mean": float(cos_sim_mean),
        "cosine_similarity_std": float(np.std(cos_sim_per_window)),
        "cosine_similarity_min": float(np.min(cos_sim_per_window)),
        "sync_score": float(sync_score),
        "alignment_score": float(alignment_score),
        "suspicious_window_fraction": float(suspicious_fraction),
        "_per_second_sync": per_second_sync,
        "_suspicious_timestamps": suspicious_timestamps,
        "_audio_events": audio_events,
        "_sync_score_raw": sync_score,
        "_active_second_count": active_seconds,
    }
    return features


# ---------------------------------------------------------------------------
# Stage 4 additions: Wasserstein distance, lip-velocity xcorr, Whisper sync
# ---------------------------------------------------------------------------

def compute_wasserstein_distance(audio_energy, lip_signal):
    # Earth Mover Distance between audio energy and lip MAR distributions.
    # Small = similar distributions (authentic). Large = decoupled (deepfake).
    try:
        from scipy.stats import wasserstein_distance as _wass
    except ImportError:
        return 0.0
    audio_energy = np.asarray(audio_energy, dtype=np.float64).ravel()
    lip_signal   = np.asarray(lip_signal,   dtype=np.float64).ravel()
    if audio_energy.size < 2 or lip_signal.size < 2:
        return 0.0
    def _norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8)
    try:
        return float(np.clip(_wass(_norm(audio_energy), _norm(lip_signal)), 0.0, 2.0))
    except Exception as exc:
        logger.warning("Wasserstein distance failed: %s", exc)
        return 0.0


def compute_lip_velocity_xcorr(lip_velocity, audio_energy, fps=25.0):
    # Cross-correlate lip velocity against audio energy.
    # Lip velocity peaks align with phoneme bursts in authentic speech.
    if lip_velocity is None or audio_energy is None:
        return 0.0, 0.0
    lv = np.asarray(lip_velocity, dtype=np.float64).ravel()
    ae = np.asarray(audio_energy, dtype=np.float64).ravel()
    if lv.size < 4 or ae.size < 4:
        return 0.0, 0.0
    return compute_cross_correlation(ae, lv, sample_rate=fps)


def compute_whisper_sync_score(whisper_info, lip_signal, fps=25.0):
    # Check whether Whisper speech segments align with lip activity.
    # Returns score in [0,1] where 1 = perfect alignment, 0.5 = neutral.
    if not whisper_info or not whisper_info.get("available"):
        return 0.5
    segments = whisper_info.get("segments") or []
    if not segments:
        return 0.5
    lip_signal = np.asarray(lip_signal, dtype=np.float64).ravel()
    if lip_signal.size < 2:
        return 0.5
    baseline      = float(np.percentile(lip_signal, 20))
    active_thresh = baseline + 0.5 * float(np.std(lip_signal))
    hit, total = 0, 0
    for seg in segments:
        try:
            sf = max(0, int(float(seg["start"]) * fps))
            ef = min(len(lip_signal), int(float(seg["end"]) * fps))
        except (KeyError, TypeError, ValueError):
            continue
        if ef <= sf:
            continue
        total += 1
        if float(np.mean(lip_signal[sf:ef])) > active_thresh:
            hit += 1
    return float(hit) / total if total > 0 else 0.5


# Capture original before rebinding -- prevents v2 -> v1 -> v2 recursion
_compute_sync_features_original = compute_sync_features


def compute_sync_features_v2(audio_energy, lip_signal, audio_events,
                               fps=25.0, lip_velocity=None, whisper_info=None):
    # Extended sync adding Wasserstein, lip-velocity xcorr, Whisper sync.
    base    = _compute_sync_features_original(audio_energy, lip_signal, audio_events, fps)
    a_arr   = np.asarray(audio_energy, dtype=np.float64).ravel()
    l_arr   = np.asarray(lip_signal,   dtype=np.float64).ravel()
    min_len = min(a_arr.size, l_arr.size)

    wass_dist  = compute_wasserstein_distance(a_arr[:min_len], l_arr[:min_len])
    wass_score = float(np.exp(-wass_dist * 4.0))

    lv_xcorr, lv_lag = 0.0, 0.0
    if lip_velocity is not None and len(lip_velocity) > 0:
        lv_arr = np.asarray(lip_velocity, dtype=np.float64).ravel()
        if lv_arr.size != min_len and min_len > 0:
            lv_arr = np.interp(
                np.linspace(0, 1, min_len),
                np.linspace(0, 1, lv_arr.size),
                lv_arr,
            )
        lv_xcorr, lv_lag = compute_lip_velocity_xcorr(lv_arr, a_arr[:min_len], fps)

    whisper_sync = compute_whisper_sync_score(whisper_info or {}, l_arr, fps)

    if base.get("_active_second_count", 0) > 0:
        xcorr    = base["xcorr_peak"]
        cos_mean = base["cosine_similarity_mean"]
        dtw_sc   = base["dtw_score"]
        if bool((whisper_info or {}).get("available")):
            new_sync = (0.30 * xcorr + 0.25 * cos_mean + 0.15 * dtw_sc
                        + 0.15 * wass_score + 0.10 * max(0.0, float(lv_xcorr))
                        + 0.05 * whisper_sync)
        else:
            new_sync = (0.35 * xcorr + 0.30 * cos_mean
                        + 0.20 * dtw_sc + 0.15 * wass_score)
        base["sync_score"] = float(np.clip(new_sync, 0.0, 1.0))

    base["wasserstein_distance"] = round(float(wass_dist),   4)
    base["wasserstein_score"]    = round(float(wass_score),  4)
    base["lip_velocity_xcorr"]   = round(float(lv_xcorr),    4)
    base["lip_velocity_lag_abs"] = round(float(abs(lv_lag)), 4)
    base["whisper_sync_score"]   = round(float(whisper_sync), 4)
    return base


# Override public name
compute_sync_features = compute_sync_features_v2