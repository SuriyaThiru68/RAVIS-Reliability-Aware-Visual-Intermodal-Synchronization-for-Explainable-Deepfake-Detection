"""
video_processor.py
Handles video validation, frame extraction, and audio extraction.
"""

import os
import cv2
import numpy as np
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_DURATION_SECONDS = 300   # 5 minutes max
MAX_FILE_SIZE_MB = 500       # match frontend 500 MB limit
TARGET_FPS = 25
TARGET_FRAME_SIZE = (256, 256)


def validate_video(filepath: str) -> dict:
    """
    Validate a video file and return metadata.
    Returns dict with keys: valid, error, duration, fps, frame_count, width, height

    Note: WebM files from MediaRecorder often report frame_count=0 in OpenCV headers.
    We fall back to reading actual frames to compute real duration.
    """
    if not filepath:
        return {"valid": False, "error": "No video file path provided"}

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        return {"valid": False, "error": f"Unsupported format: {ext}"}

    if not os.path.exists(filepath):
        return {"valid": False, "error": f"Video file not found: {filepath}"}

    try:
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    except OSError:
        return {"valid": False, "error": "Could not read video file"}

    if file_size_mb > MAX_FILE_SIZE_MB:
        return {"valid": False, "error": f"File too large: {file_size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)"}

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return {"valid": False, "error": "Could not open video file"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MediaRecorder WebM files frequently report fps=1000 (or 0) and
    # frame_count=0. Treat implausible values as unknown, otherwise the
    # duration comes out ~30x too short and valid clips are rejected.
    if fps <= 0 or fps > 240:
        if fps > 0:
            logger.info(f"Implausible container fps={fps:.1f} for "
                        f"{os.path.basename(filepath)}; ignoring it")
        fps = 0.0

    # WebM / MediaRecorder files often report frame_count=0 or wrong duration.
    # Use CAP_PROP_POS_MSEC by seeking to end, or count actual frames as fallback.
    duration = 0.0
    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
    else:
        # Try OpenCV duration via millisecond property
        try:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if duration_ms > 0:
                duration = duration_ms / 1000.0
                frame_count = int(duration * max(fps, 25.0))
        except Exception:
            pass

    # Still zero — count frames directly (slower but reliable for short clips).
    # Read the per-frame timestamp as we go; some WebM writers expose it even
    # when the header duration/fps are missing.
    if duration <= 0 or fps <= 0:
        logger.info(f"Counting frames directly for {os.path.basename(filepath)} (WebM fallback)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        actual_frames = 0
        last_msec = 0.0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            actual_frames += 1
            try:
                msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if msec > last_msec:
                    last_msec = msec
            except Exception:
                pass
            if actual_frames > 10000:  # safety cap
                break

        assumed_fps = fps if 0 < fps <= 240 else 30.0
        frame_count = actual_frames
        if last_msec > 0 and last_msec / 1000.0 >= 0.5:
            # Timestamps are trustworthy — derive both duration and real fps.
            duration = last_msec / 1000.0
            fps = actual_frames / duration if duration > 0 else assumed_fps
        else:
            duration = actual_frames / assumed_fps
            fps = assumed_fps

    cap.release()

    if fps <= 0:
        fps = 25.0

    if duration > MAX_DURATION_SECONDS:
        return {"valid": False, "error": f"Video too long: {duration:.1f}s (max {MAX_DURATION_SECONDS}s)"}

    # Accept videos as short as 0.5 seconds (live recordings can be brief)
    if duration < 0.5:
        return {"valid": False, "error": f"Video too short ({duration:.2f}s). Please record at least 1 second."}

    return {
        "valid": True,
        "error": None,
        "duration": round(duration, 2),
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "file_size_mb": round(file_size_mb, 2),
    }


def extract_frames(filepath: str, max_frames: int = 1200,
                   duration_hint: float = None) -> tuple:
    """
    Extract frames uniformly across the clip, at most ``max_frames`` frames
    and never more than TARGET_FPS.

    Returns ``(frames, original_fps, sampling_fps)`` where ``sampling_fps`` is
    the *true* rate of the returned frames on the source timeline.  Every
    downstream stage must use ``sampling_fps`` — assuming 25 fps when the
    container is 24/30 fps (or when frames are decimated to fit the cap) shifts
    the lip signal's timebase and corrupts every synchronisation metric.
    """
    if not filepath or not os.path.exists(filepath):
        logger.warning(f"Video file not found for frame extraction: {filepath}")
        return [], 0.0, 0.0

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        logger.warning(f"Could not open video for frame extraction: {filepath}")
        return [], 0.0, 0.0

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0 or original_fps > 240:
        original_fps = 30.0  # MediaRecorder WebM commonly claims 1000 fps

    # Determine how far the clip runs so the sampling interval can be chosen
    # to cover the *whole* video instead of silently truncating it.
    duration = float(duration_hint) if duration_hint else 0.0
    if duration <= 0:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            duration = frame_count / original_fps

    if duration > 0:
        # Aim for the target rate, but decimate further for long clips so the
        # frame budget covers the complete duration.
        desired_rate = min(float(TARGET_FPS), max_frames / duration)
        sample_interval = max(1, int(np.ceil(original_fps / max(desired_rate, 1e-6))))
    else:
        sample_interval = max(1, int(round(original_fps / TARGET_FPS)))

    sampling_fps = original_fps / sample_interval
    if sampling_fps <= 0:
        sampling_fps = original_fps

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            resized = cv2.resize(frame, TARGET_FRAME_SIZE)
            frames.append(resized)
            if len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    logger.info(
        f"Extracted {len(frames)} frames, total_read={frame_idx}, "
        f"interval={sample_interval}, container_fps={original_fps:.2f}, "
        f"sampling_fps={sampling_fps:.2f}, covered={len(frames)/max(sampling_fps,1e-6):.2f}s"
    )
    return frames, original_fps, sampling_fps


def _get_ffmpeg_exe() -> str:
    """Return the ffmpeg executable path — uses bundled imageio_ffmpeg if system ffmpeg not found."""
    import shutil
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"   # last resort, will fail gracefully


def extract_audio(video_filepath: str, output_dir: str):
    """
    Extract audio from video using ffmpeg.
    Returns path to the extracted WAV file, or None on failure.
    Output: 16kHz, mono, PCM 16-bit WAV
    """
    audio_path = os.path.join(output_dir, "audio.wav")

    cmd = [
        _get_ffmpeg_exe(),
        "-y",
        "-i", video_filepath,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz sample rate
        "-acodec", "pcm_s16le",
        "-vn",               # no video
        audio_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr.decode()}")
            # Try fallback: silent audio scenario
            return _create_silent_audio(audio_path)
        logger.info(f"Audio extracted to {audio_path}")
        return audio_path
    except FileNotFoundError:
        logger.warning("ffmpeg not found in PATH. Trying bundled imageio_ffmpeg.")
        return _extract_audio_moviepy(video_filepath, audio_path)
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out")
        return None


def _extract_audio_moviepy(video_filepath: str, output_path: str):
    """Fallback audio extraction using moviepy."""
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_filepath)
        if clip.audio is None:
            logger.warning("Video has no audio track")
            clip.close()
            return _create_silent_audio(output_path)
        clip.audio.write_audiofile(output_path, fps=16000, nbytes=2, ffmpeg_params=["-ac", "1"], logger=None)
        clip.close()
        return output_path
    except Exception as e:
        logger.error(f"moviepy extraction failed: {e}")
        return _create_silent_audio(output_path)


def _create_silent_audio(output_path: str, duration: float = 10.0, sr: int = 16000):
    """Create a silent audio file as fallback for videos without audio."""
    try:
        import scipy.io.wavfile as wav
        silent = np.zeros(int(sr * duration), dtype=np.int16)
        wav.write(output_path, sr, silent)
        logger.warning(f"Created silent audio fallback at {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Could not create silent audio: {e}")
        return None


def get_frame_timestamps(frame_count: int, fps: float = TARGET_FPS) -> np.ndarray:
    """Return an array of timestamps (in seconds) for each frame."""
    return np.arange(frame_count) / fps
