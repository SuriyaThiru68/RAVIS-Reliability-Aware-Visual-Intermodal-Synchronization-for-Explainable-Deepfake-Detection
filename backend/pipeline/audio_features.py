"""
audio_features.py
Extracts MFCC, Mel-spectrogram, and voice quality features from WAV audio.
Also computes the audio energy envelope (phoneme burst signal).
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Audio processing constants
SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 64
HOP_LENGTH = 512        # ~32ms at 16kHz
N_FFT = 1024
FRAME_DURATION = HOP_LENGTH / SAMPLE_RATE  # seconds per frame


def load_audio(wav_path: str) -> tuple[np.ndarray, int]:
    """
    Load a WAV file using librosa.
    Returns (audio_array, sample_rate)
    """
    try:
        import librosa
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        logger.info(f"Loaded audio: {len(audio)/sr:.2f}s, sr={sr}")
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        # Return 10s of silence as fallback
        return np.zeros(SAMPLE_RATE * 10, dtype=np.float32), SAMPLE_RATE


def extract_mfcc_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Extract MFCC, delta, and delta-delta features.
    Returns dict with statistical summaries (mean, std, min, max) per coefficient.
    """
    import librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = {}
    for i in range(N_MFCC):
        for stat, fn in [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
            features[f"mfcc_{i}_{stat}"] = float(fn(mfcc[i]))
            features[f"mfcc_d_{i}_{stat}"] = float(fn(delta[i]))

    features["mfcc_mean_global"] = float(np.mean(mfcc))
    features["mfcc_std_global"] = float(np.std(mfcc))
    return features


def extract_mel_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Extract Mel spectrogram statistics.
    """
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    features = {
        "mel_mean": float(np.mean(mel_db)),
        "mel_std": float(np.std(mel_db)),
        "mel_max": float(np.max(mel_db)),
        "mel_min": float(np.min(mel_db)),
        "mel_low_energy": float(np.mean(mel_db[:16])),   # Low-freq energy
        "mel_mid_energy": float(np.mean(mel_db[16:40])), # Mid-freq energy
        "mel_high_energy": float(np.mean(mel_db[40:])),  # High-freq energy
        "mel_spectral_contrast": float(np.mean(np.max(mel_db, axis=1) - np.min(mel_db, axis=1))),
    }
    return features


def extract_voice_quality_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Extract basic voice quality features:
    ZCR, RMS energy, spectral centroid, rolloff, bandwidth.
    """
    import librosa
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Silence ratio
    silence_threshold = 0.01
    silence_ratio = float(np.mean(rms < silence_threshold))

    # Speech activity (RMS > threshold)
    speech_activity = float(np.mean(rms > silence_threshold))

    features = {
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_max": float(np.max(rms)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "silence_ratio": silence_ratio,
        "speech_activity": speech_activity,
        "dynamic_range_db": float(20 * np.log10(np.max(rms) / (np.mean(rms) + 1e-6) + 1e-6)),
    }
    return features


def compute_energy_envelope(audio: np.ndarray, sr: int = SAMPLE_RATE,
                            fps: float = 25.0) -> np.ndarray:
    """
    Compute audio energy envelope resampled to video frame rate.
    This is the key signal for audio-visual synchronization.
    Returns array of shape (n_video_frames,) with energy per frame.
    """
    if audio is None:
        return np.array([], dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return np.array([], dtype=np.float32)

    import librosa
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    if rms.size == 0:
        return np.array([], dtype=np.float32)

    audio_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
    duration = len(audio) / sr if sr > 0 else 0.0

    n_video_frames = int(duration * fps)
    if n_video_frames <= 0:
        return np.array([], dtype=np.float32)

    video_times = np.linspace(0, duration, n_video_frames)
    energy = np.interp(video_times, audio_times, rms)

    e_max = np.max(energy)
    if e_max > 0:
        energy = energy / e_max

    return energy.astype(np.float32)


def compute_audio_events(energy: np.ndarray, fps: float = 25.0,
                         threshold: float = 0.15) -> list[float]:
    """
    Detect audio onset events (speech bursts) from the energy envelope.
    Returns list of timestamps (seconds) where audio events start.
    """
    if energy is None:
        return []

    energy = np.asarray(energy, dtype=np.float64)
    if energy.size == 0:
        return []

    events = []
    above = energy > threshold
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            events.append(i / max(fps, 1e-6))
    return events


def extract_all_audio_features(wav_path: str) -> dict:
    """
    Master function: extract all audio features from a WAV file.
    Returns a flat dict of features + raw energy envelope.
    """
    audio, sr = load_audio(wav_path)

    features = {}
    features.update(extract_mfcc_features(audio, sr))
    features.update(extract_mel_features(audio, sr))
    features.update(extract_voice_quality_features(audio, sr))

    # Raw signals for sync analysis
    features["_energy_envelope"] = compute_energy_envelope(audio, sr)
    features["_audio_events"] = compute_audio_events(features["_energy_envelope"])
    features["_audio_duration"] = len(audio) / sr

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 additions: SNR estimation + Whisper speech timestamps
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Estimate Signal-to-Noise Ratio (dB) using a simple noise-floor method.

    Splits the track into short frames; frames whose RMS is below the
    20th-percentile RMS are treated as noise-only segments.  Returns
    10 * log10(signal_power / noise_power).  Returns 0.0 if the audio is
    too short or completely silent.
    """
    if audio is None or len(audio) < sr // 4:
        return 0.0
    try:
        frame_len = int(sr * 0.032)   # 32 ms
        step      = frame_len // 2
        frames    = [audio[i:i + frame_len]
                     for i in range(0, len(audio) - frame_len, step)]
        rms_vals  = np.array([np.sqrt(np.mean(f ** 2)) for f in frames],
                             dtype=np.float64)
        rms_vals  = rms_vals[rms_vals > 1e-10]
        if rms_vals.size < 4:
            return 0.0
        noise_floor = float(np.percentile(rms_vals, 20))
        signal_rms  = float(np.percentile(rms_vals, 80))
        if noise_floor < 1e-10:
            return 40.0   # effectively silent noise → high SNR
        snr = 10.0 * np.log10((signal_rms / noise_floor) ** 2)
        return float(np.clip(snr, -10.0, 60.0))
    except Exception as e:
        logger.warning("SNR computation failed: %s", e)
        return 0.0


def extract_whisper_timestamps(wav_path: str) -> dict:
    """
    Use OpenAI Whisper to produce word/segment-level speech timestamps.

    Returns a dict::

        {
          "word_timestamps": [{"word": "hello", "start": 0.12, "end": 0.42}, ...],
          "segments":        [{"text": "...", "start": 0.0, "end": 2.1}, ...],
          "language":        "en",
          "available":       True,
        }

    Falls back gracefully if Whisper is not installed or the model cannot
    be loaded, returning ``{"available": False}``.
    """
    try:
        import whisper as _whisper  # openai-whisper
    except ImportError:
        logger.warning("openai-whisper not installed — skipping Whisper timestamps")
        return {"available": False}

    try:
        # Use the tiny model so it is fast even on CPU; load once per process
        # by caching on the function object.
        if not hasattr(extract_whisper_timestamps, "_model"):
            logger.info("Loading Whisper tiny model (first use)…")
            extract_whisper_timestamps._model = _whisper.load_model("tiny")
        model = extract_whisper_timestamps._model

        result = model.transcribe(
            wav_path,
            word_timestamps=True,
            language=None,   # auto-detect
            verbose=False,
            fp16=False,      # always safe on CPU
        )

        word_ts = []
        for seg in result.get("segments") or []:
            for w in seg.get("words") or []:
                word_ts.append({
                    "word":  str(w.get("word", "")).strip(),
                    "start": float(w.get("start", 0.0)),
                    "end":   float(w.get("end",   0.0)),
                })

        segments = [
            {"text": str(s.get("text", "")),
             "start": float(s.get("start", 0.0)),
             "end":   float(s.get("end",   0.0))}
            for s in result.get("segments") or []
        ]
        logger.info(
            "Whisper: %d words / %d segments, lang=%s",
            len(word_ts), len(segments), result.get("language", "?"),
        )
        return {
            "available":       True,
            "word_timestamps": word_ts,
            "segments":        segments,
            "language":        result.get("language", ""),
        }
    except Exception as exc:
        logger.warning("Whisper transcription failed: %s", exc)
        return {"available": False}


def extract_all_audio_features_v2(wav_path: str) -> dict:
    """
    Extended version of extract_all_audio_features that also computes:
      - SNR estimation
      - Whisper word-level speech timestamps

    Replaces the original function; the original name is kept as an alias for
    backwards compatibility.
    """
    audio, sr = load_audio(wav_path)

    features = {}
    features.update(extract_mfcc_features(audio, sr))
    features.update(extract_mel_features(audio, sr))
    features.update(extract_voice_quality_features(audio, sr))

    # SNR — used by reliability scoring
    features["snr_db"] = compute_snr(audio, sr)

    # Raw signals for sync analysis (unchanged)
    features["_energy_envelope"]   = compute_energy_envelope(audio, sr)
    features["_audio_events"]      = compute_audio_events(features["_energy_envelope"])
    features["_audio_duration"]    = len(audio) / sr

    # Whisper timestamps — stored as private key; passed to sync_analyzer
    features["_whisper"]           = extract_whisper_timestamps(wav_path)

    return features


# Alias so existing callers (app.py) can switch to the extended version
# without a separate code change.
extract_all_audio_features = extract_all_audio_features_v2
