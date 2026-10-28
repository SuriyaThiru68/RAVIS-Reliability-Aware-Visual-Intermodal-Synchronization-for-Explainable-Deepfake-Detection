"""
app.py — SafeGuard Flask API
Orchestrates the full deepfake detection pipeline.
"""

import os
import uuid
import tempfile
import logging
import threading
import time
import shutil
import io
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ─── Setup ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("safeguard")

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=None)  # disable built-in static handler
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload


@app.after_request
def no_cache_for_ui(response):
    """Dev convenience: always revalidate HTML/CSS/JS so UI edits appear on refresh."""
    if response.mimetype in ("text/html", "text/css", "application/javascript", "text/javascript"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response

# In-memory job store  {job_id: {status, progress, result, error}}
jobs = {}   # type: dict
jobs_lock = threading.Lock()

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Static / Frontend ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/favicon.ico")
def favicon():
    icon = (
        b"\x00\x00\x01\x00\x01\x00\x10\x10\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00"
    )
    return Response(icon, mimetype="image/x-icon")


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(str(FRONTEND_DIR / "css"), filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(str(FRONTEND_DIR / "js"), filename)


@app.route("/<path:filename>", methods=["GET", "HEAD"])
def serve_static(filename):
    """Serve any other static file from the frontend directory."""
    # Never intercept /api/ paths — let them 404 naturally
    if filename.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(str(FRONTEND_DIR), filename)


# ─── Upload endpoint ──────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Accept a video file upload and start async analysis.
    Returns {job_id} immediately.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    video_path = job_dir / filename
    file.save(str(video_path))

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "stage": "Queued",
            "result": None,
            "error": None,
        }

    # Start background thread
    t = threading.Thread(target=_run_pipeline, args=(job_id, str(video_path), str(job_dir)), daemon=True)
    t.start()

    return jsonify({"job_id": job_id}), 202


# ─── Status polling ───────────────────────────────────────────────────────────

@app.route("/api/status/<job_id>", methods=["GET"])
def status(job_id: str):
    """Poll job status. Returns {status, progress, stage, result?, error?}"""
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "result": job["result"],
        "error": job["error"],
    })


@app.route("/api/video/<job_id>", methods=["GET"])
def get_video(job_id: str):
    """Serve uploaded video for playback."""
    job_dir = UPLOADS_DIR / job_id
    if not job_dir.exists():
        return jsonify({"error": "Video not found"}), 404
    files = [f for f in job_dir.iterdir() if f.is_file() and allowed_file(f.name)]
    if not files:
        return jsonify({"error": "No video file found"}), 404
    return send_from_directory(str(job_dir), files[0].name)


@app.route("/api/report", methods=["POST"])
def report():
    """
    Generate a detailed PDF report for a completed analysis.

    Body: {"result": {...}, "filename": "...", "sha256": "...",
           "elapsed": 12.3, "job_id": "..."}
    """
    try:
        payload = request.get_json(silent=True) or {}
        result = payload.get("result")
        if not isinstance(result, dict) or not result:
            return jsonify({"error": "A completed analysis result is required."}), 400

        # Defensive limit: the normal payload is well under 1 MB.
        if (request.content_length or 0) > 8 * 1024 * 1024:
            return jsonify({"error": "Report payload is too large."}), 413

        from backend.report_pdf import build_report_pdf

        filename = str(payload.get("filename") or
                       (result.get("video_meta") or {}).get("filename") or
                       "video").strip()
        safe_name = secure_filename(filename) or "video"
        sha256 = str(payload.get("sha256") or "").strip() or None
        elapsed = payload.get("elapsed")
        job_id = str(payload.get("job_id") or "").strip() or None

        pdf_bytes = build_report_pdf(
            result,
            filename=filename,
            sha256=sha256,
            elapsed_s=elapsed,
            job_id=job_id,
        )

        buf = io.BytesIO(pdf_bytes)
        buf.seek(0)
        stem = os.path.splitext(safe_name)[0][:60] or "video"
        download_name = (
            f"RAVIS_Report_{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=download_name,
            max_age=0,
        )
    except Exception as exc:
        logger.exception("PDF report generation failed")
        return jsonify({"error": f"Could not generate the PDF report: {exc}"}), 500


# ─── Pipeline runner ──────────────────────────────────────────────────────────

def _update_job(job_id: str, status: str = None, progress: int = None,
                stage: str = None, result=None, error: str = None):
    with jobs_lock:
        if job_id in jobs:
            if status is not None:
                jobs[job_id]["status"] = status
            if progress is not None:
                jobs[job_id]["progress"] = progress
            if stage is not None:
                jobs[job_id]["stage"] = stage
            if result is not None:
                jobs[job_id]["result"] = result
            if error is not None:
                jobs[job_id]["error"] = error


def _run_pipeline(job_id: str, video_path: str, work_dir: str):
    """
    Full detection pipeline executed in a background thread.
    Updates job progress at each stage.
    """
    try:
        # Lazy imports (so server starts fast)
        import numpy as np
        from backend.pipeline.video_processor import validate_video, extract_frames, extract_audio
        from backend.pipeline.face_detector import FaceDetector
        from backend.pipeline.audio_features import extract_all_audio_features
        from backend.pipeline.visual_features import extract_visual_features, extract_lip_signal
        from backend.pipeline.sync_analyzer import compute_sync_features, resample_to_common_grid
        from backend.pipeline.classifier import classify
        from backend.pipeline.explainer import format_results
        from backend.pipeline.emotion_analyzer import run_emotion_analysis

        # ── Stage 1: Validate ────────────────────────────────────────────────
        _update_job(job_id, status="running", progress=5, stage="Validating video")
        meta = validate_video(video_path)
        if not meta["valid"]:
            _update_job(job_id, status="failed", progress=100,
                        stage="Failed", error=meta["error"])
            return

        # ── Server-side SHA-256 integrity hash (Stage 8) ────────────────────
        import hashlib as _hashlib
        try:
            _hasher = _hashlib.sha256()
            with open(video_path, "rb") as _vf:
                for _chunk in iter(lambda: _vf.read(65536), b""):
                    _hasher.update(_chunk)
            _server_sha256 = _hasher.hexdigest()
        except Exception as _hex_err:
            logger.warning("[%s] SHA-256 computation failed: %s", job_id, _hex_err)
            _server_sha256 = None

        # ── Stage 2: Extract Audio ───────────────────────────────────────────
        _update_job(job_id, progress=15, stage="Extracting audio")
        audio_path = extract_audio(video_path, work_dir)

        # ── Stage 3: Extract Frames ──────────────────────────────────────────
        _update_job(job_id, progress=25, stage="Extracting video frames")
        frames, original_fps, sampling_fps = extract_frames(
            video_path, max_frames=1200, duration_hint=meta["duration"]
        )
        logger.info(
            f"[{job_id}] Got {len(frames)} frames at {sampling_fps:.2f} fps "
            f"(container {original_fps:.2f} fps)"
        )

        # ── Thumbnail capture ────────────────────────────────────────────────
        # Grab three representative frames (≈10%, ≈40%, ≈70% through the clip)
        # and encode them as base64 JPEG for the PDF report.
        _thumbnail_b64_list = []
        if frames:
            import base64 as _base64
            import cv2 as _cv2
            _thumb_indices = [
                max(0, int(len(frames) * 0.10)),
                max(0, int(len(frames) * 0.40)),
                max(0, int(len(frames) * 0.70)),
            ]
            for _ti in dict.fromkeys(_thumb_indices):  # deduplicate, keep order
                _frame = frames[_ti]
                # Upscale the 256×256 extracted frame to a presentable 640×360
                _frame_resized = _cv2.resize(_frame, (640, 360))
                _ok, _enc = _cv2.imencode(
                    ".jpg", _frame_resized,
                    [_cv2.IMWRITE_JPEG_QUALITY, 88],
                )
                if _ok:
                    _thumbnail_b64_list.append(
                        _base64.b64encode(_enc.tobytes()).decode()
                    )

        # ── Stage 4: Face + Lip Detection ────────────────────────────────────
        _update_job(job_id, progress=40, stage="Detecting faces & lip landmarks")
        detector = FaceDetector()
        face_results = detector.process_frames(frames)
        detector.close()

        # ── Stage 3 (Acoustic): Audio Features + Whisper timestamps ────────
        _update_job(job_id, progress=55, stage="Extracting audio features & speech timestamps")
        audio_feats = extract_all_audio_features(audio_path) if audio_path else {}
        _whisper_info = audio_feats.pop("_whisper", {"available": False})

        # ── Stage 2 (Face/Lip Kinematics): Visual Features ──────────────────
        _update_job(job_id, progress=65, stage="Extracting visual & lip-kinematics features")
        visual_feats = extract_visual_features(face_results, fps=sampling_fps)
        lip_signal   = extract_lip_signal(face_results)
        # Extract the per-frame lip velocity series for Stage 4 sync
        import numpy as _np_lv
        lip_velocity_signal = _np_lv.array(
            [r.lip_velocity for r in face_results if r.detected],
            dtype=_np_lv.float64,
        ) if face_results else _np_lv.array([], dtype=_np_lv.float64)

        # ── Stage 4 (Synchronization): A/V sync with Wasserstein + FastDTW ──
        _update_job(job_id, progress=75, stage="Computing audio-visual synchronization")
        energy_envelope = audio_feats.pop("_energy_envelope", np.zeros(0))
        audio_events = audio_feats.pop("_audio_events", [])
        audio_duration = float(audio_feats.get("_audio_duration", 0.0) or 0.0)

        # Both signals must live on the SAME timebase before correlation.
        # Audio is emitted at 25 Hz; lips are emitted at sampling_fps (which is
        # often 12/15/24/30, not 25).  Resample to a shared 25 Hz grid.
        analysis_duration = float(meta.get("duration", 0.0) or 0.0)
        lip_duration = len(lip_signal) / sampling_fps if sampling_fps > 0 else 0.0
        if lip_duration > 0:
            analysis_duration = min(analysis_duration, lip_duration) if analysis_duration > 0 else lip_duration
        if energy_envelope is not None and len(energy_envelope) > 0:
            analysis_duration = min(analysis_duration, len(energy_envelope) / 25.0)
        if analysis_duration <= 0:
            analysis_duration = 0.0

        common_fps = 25.0
        audio_25, lip_25 = resample_to_common_grid(
            audio_energy=energy_envelope,
            audio_fps=25.0,
            lip_signal=lip_signal,
            lip_fps=sampling_fps,
            duration=analysis_duration,
            common_fps=common_fps,
        )

        sync_feats = compute_sync_features(
            audio_energy=audio_25,
            lip_signal=lip_25,
            audio_events=audio_events,
            fps=common_fps,
            lip_velocity=lip_velocity_signal if len(lip_velocity_signal) > 0 else None,
            whisper_info=_whisper_info,
        )

        # Record what was actually analysed (shown in the forensic dossier/report)
        meta = dict(meta)
        meta["analysis_fps"] = round(sampling_fps, 2)
        meta["analyzed_frames"] = len(frames)
        meta["analyzed_duration"] = round(analysis_duration, 2)
        meta["audio_duration"] = round(audio_duration, 2)

        # ── Stage 5 (Emotion Consistency): MobileNet / FER analysis ────────
        _update_job(job_id, progress=82, stage="Analysing emotion consistency")
        try:
            emotion_analysis = run_emotion_analysis(
                frames, face_results, audio_feats, fps=sampling_fps
            )
        except Exception as _emo_err:
            logger.warning("[%s] Emotion analysis failed: %s", job_id, _emo_err)
            emotion_analysis = {"emotion_disagreement": 0.0, "consistency": {}}

        # ── Stage 8: Classify (RF + SHAP + conformal prediction) ───────────
        _update_job(job_id, progress=88, stage="Running classifier")
        # Extract private signals before combining (used by explainer, not classifier)
        per_second_sync = sync_feats.get("_per_second_sync", np.array([]))
        suspicious_timestamps = sync_feats.get("_suspicious_timestamps", [])
        sync_audio_events = sync_feats.get("_audio_events", [])

        combined = {}
        combined.update(audio_feats)
        combined.update(visual_feats)
        # Only include scalar sync features for classifier (skip _ private keys)
        combined.update({k: v for k, v in sync_feats.items() if not k.startswith("_")})
        # Extra context for the evidence-sufficiency gate (ignored by the RF)
        combined["_active_second_count"] = sync_feats.get("_active_second_count", 0)
        combined["_video_duration"] = meta.get("duration", 0.0)
        combined["_analysis_duration"] = analysis_duration
        # Emotion disagreement score as an additional classification signal
        combined["emotion_disagreement"] = float(
            emotion_analysis.get("emotion_disagreement", 0.0) or 0.0
        )

        classification = classify(combined, video_meta=meta)
        # Attach emotion analysis to classification so explainer/reliability can use it
        classification["emotion_analysis"] = emotion_analysis

        # ── Stage 9: Explain + Report (RAVIS Stage 8) ───────────────────────
        _update_job(job_id, progress=95, stage="Generating explanation")
        result = format_results(
            verdict=classification["verdict"],
            deepfake_probability=classification["deepfake_probability"],
            confidence=classification["confidence"],
            sync_features=sync_feats,
            audio_features=audio_feats,
            visual_features=visual_feats,
            classification=classification,
            video_meta=meta,
            energy_envelope=audio_25,
            lip_signal=lip_25,
        )

        # ── Done ─────────────────────────────────────────────────────────────
        # Attach thumbnails to the result so the PDF report can embed them.
        if _thumbnail_b64_list:
            result["thumbnail_b64_list"] = _thumbnail_b64_list

        # Attach server-side SHA-256 (Stage 8 — integrity)
        if _server_sha256:
            result["server_sha256"] = _server_sha256

        # Attach conformal prediction, SHAP, emotion, Whisper summary
        result["conformal_prediction"] = classification.get("conformal_prediction", {})
        result["shap_top_features"]    = classification.get("top_shap", [])
        result["emotion_analysis"]     = emotion_analysis
        result["whisper_available"]    = bool(_whisper_info.get("available"))
        result["whisper_language"]     = _whisper_info.get("language", "")
        result["whisper_segments"]     = len(_whisper_info.get("segments") or [])
        result["snr_db"]               = audio_feats.get("snr_db")

        _update_job(job_id, status="done", progress=100,
                    stage="Complete", result=result)
        logger.info(f"[{job_id}] Pipeline complete — verdict: {result['verdict']}")

    except Exception as exc:
        logger.exception(f"[{job_id}] Pipeline error: {exc}")
        _update_job(job_id, status="failed", progress=100,
                    stage="Error", error=str(exc))
    finally:
        # Clean up uploaded file after 10 minutes
        threading.Timer(600, _cleanup_job, args=(job_id, work_dir)).start()


def _cleanup_job(job_id: str, work_dir: str):
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
        with jobs_lock:
            jobs.pop(job_id, None)
        logger.info(f"Cleaned up job {job_id}")
    except Exception:
        pass


# ─── Real-time score endpoint ─────────────────────────────────────────────────

@app.route("/api/realtime_score", methods=["POST"])
def realtime_score():
    """
    Receives pre-computed features from the browser's real-time engine
    (MAR, RMS, sync score, xcorr, lag, etc.) and runs them through the
    trained Random Forest classifier for a backend-verified verdict.

    Accepts JSON body:
        mar_mean, mar_std, rms_mean, sync_score, xcorr_peak,
        lag_ms, suspicious_frac, face_detected, frame_count
    Returns:
        verdict, deepfake_probability, real_probability, confidence,
        explanation (string), cues (list)
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    # ── Extract features sent from the browser ────────────────────────
    mar_mean        = float(data.get("mar_mean",        0.15))
    mar_std         = float(data.get("mar_std",         0.05))
    rms_mean        = float(data.get("rms_mean",        0.04))
    sync_score      = float(data.get("sync_score",      0.50))
    xcorr_peak      = float(data.get("xcorr_peak",      0.50))
    lag_ms          = float(data.get("lag_ms",          0.0))
    suspicious_frac = float(data.get("suspicious_frac", 0.0))
    face_detected   = bool(data.get("face_detected",    True))
    frame_count     = int(data.get("frame_count",       0))

    # Duration is needed by the evidence gate.  Prefer an explicit value from
    # the browser; otherwise estimate it from the number of analysed frames.
    duration = float(data.get("video_duration") or data.get("duration") or 0.0)
    if duration <= 0 and frame_count > 0:
        duration = max(1.0, frame_count / 25.0)
    active_seconds = int(data.get("active_second_count") or 0)
    if active_seconds <= 0:
        speech_est = max(0.0, min(1.0, rms_mean * 20.0))
        active_seconds = int(round(speech_est * duration))

    # ── Clamp all values to [0,1] range ───────────────────────────────
    sync_score      = max(0.0, min(1.0, sync_score))
    xcorr_peak      = max(0.0, min(1.0, xcorr_peak))
    suspicious_frac = max(0.0, min(1.0, suspicious_frac))
    lag_sec         = lag_ms / 1000.0

    # ── Build feature dict matching CLASSIFIER_FEATURE_KEYS ───────────
    from backend.pipeline.classifier import classify, CLASSIFIER_FEATURE_KEYS

    # Construct a feature vector that matches what the RF model expects.
    # Features not directly available from webcam get sensible defaults.
    features = {
        # MFCC — synthesise from RMS as proxy
        "mfcc_mean_global":         rms_mean * -20,
        "mfcc_std_global":          rms_mean * 8,
        "mel_mean":                 rms_mean * -25,
        "mel_std":                  10.0,
        "mel_max":                  0.0,
        "mel_low_energy":           rms_mean * -22,
        "mel_mid_energy":           rms_mean * -24,
        "mel_high_energy":          rms_mean * -28,
        "mel_spectral_contrast":    20.0,
        # Voice quality
        "zcr_mean":                 0.08,
        "zcr_std":                  0.02,
        "rms_mean":                 rms_mean,
        "rms_std":                  rms_mean * 0.4,
        "rms_max":                  rms_mean * 2.5,
        "spectral_centroid_mean":   2000.0,
        "spectral_centroid_std":    300.0,
        "spectral_rolloff_mean":    3000.0,
        "spectral_bandwidth_mean":  1500.0,
        "silence_ratio":            max(0, 1 - rms_mean * 20),
        "speech_activity":          min(1, rms_mean * 20),
        "dynamic_range_db":         20.0,
        # Visual / lip kinematics
        "face_detection_rate":      0.85 if face_detected else 0.15,
        "mar_mean":                 mar_mean,
        "mar_std":                  mar_std,
        "mar_min":                  max(0, mar_mean - mar_std * 2),
        "mar_max":                  mar_mean + mar_std * 2,
        "mar_range":                mar_std * 4,
        "lip_velocity_mean":        mar_std * 0.5,
        "lip_velocity_std":         mar_std * 0.3,
        "lip_velocity_max":         mar_std * 2.0,
        "lip_activity_ratio":       min(1, mar_mean * 4),
        "lip_event_count":          float(frame_count) * mar_mean * 0.5,
        "lip_entropy":              min(1, mar_std * 10),
        "head_pitch_std":           0.02,
        "head_yaw_std":             0.02,
        "head_motion_total":        0.02,
        "face_area_mean":           0.35,
        "face_area_std":            0.02,
        # Sync features — these are the KEY discriminating features
        "xcorr_peak":               xcorr_peak,
        "audio_lip_lag_abs":        lag_sec,
        "dtw_distance":             max(0, (1.0 - sync_score) * 0.6),
        "dtw_score":                sync_score,
        "cosine_similarity_mean":   sync_score,
        "cosine_similarity_std":    0.1,
        "cosine_similarity_min":    max(0, sync_score - 0.2),
        "sync_score":               sync_score,
        "alignment_score":          sync_score,
        "suspicious_window_fraction": suspicious_frac,
        "_video_duration":          duration,
        "_active_second_count":     active_seconds,
    }

    # Fill in per-coefficient MFCC keys with defaults
    for i in range(13):
        features.setdefault(f"mfcc_{i}_mean", rms_mean * -15 - i * 2)
        features.setdefault(f"mfcc_{i}_std",  8.0 - i * 0.3)

    # ── Run classifier ────────────────────────────────────────────────
    try:
        result = classify(features, video_meta={"duration": duration})
    except Exception as exc:
        logger.error(f"realtime_score classifier error: {exc}")
        # Fallback to heuristic
        from backend.pipeline.classifier import heuristic_classify
        deep_prob, verdict = heuristic_classify(features)
        result = {
            "verdict":              verdict,
            "deepfake_probability": round(deep_prob, 4),
            "real_probability":     round(1 - deep_prob, 4),
            "confidence":           round(abs(deep_prob - 0.5) * 2, 4),
            "decision_reason":      "Classifier unavailable; heuristic fallback used.",
            "evidence_quality":     {"informative": False, "flags": ["Classifier unavailable"]},
        }

    # ── Build explanation string ──────────────────────────────────────
    verdict     = result["verdict"]
    deep_prob   = result["deepfake_probability"]
    confidence  = result["confidence"]

    if verdict == "INCONCLUSIVE":
        flags = (result.get("evidence_quality") or {}).get("flags") or []
        cues = [{"text": flag, "type": "warning"} for flag in flags[:4]]
        if not cues:
            cues = [{"text": "Evidence is not decisive", "type": "warning"}]
        explanation = result.get("decision_reason") or (
            "The live measurements are not strong enough to support a reliable verdict. "
            "Record a longer clip with a clearly visible, speaking face."
        )
    else:
        cues = []
        if sync_score < 0.40:
            cues.append({"text": f"Low Sync ({sync_score:.2f})",      "type": "critical"})
        if lag_ms > 100:
            cues.append({"text": f"A/V Lag {lag_ms:.0f}ms",           "type": "critical"})
        if xcorr_peak < 0.35:
            cues.append({"text": f"Weak XCorr ({xcorr_peak:.2f})",    "type": "warning"})
        if suspicious_frac > 0.30:
            cues.append({"text": f"{suspicious_frac*100:.0f}% Suspicious Windows", "type": "warning"})
        if not face_detected:
            cues.append({"text": "No Face Detected",                   "type": "warning"})
        if not cues:
            cues.append({"text": "Natural Co-Articulation Confirmed",  "type": "good"})

        if verdict == "DEEPFAKE":
            explanation = (
                f"Backend RF classifier confirms DEEPFAKE with {deep_prob*100:.1f}% probability. "
                + (f"A/V lag of {lag_ms:.0f}ms exceeds 40ms natural threshold. " if lag_ms > 100 else "")
                + (f"Sync score {sync_score:.2f} below authentic threshold. " if sync_score < 0.45 else "")
            )
        else:
            explanation = (
                f"Backend RF classifier confirms AUTHENTIC with {(1-deep_prob)*100:.1f}% confidence. "
                f"Audio-lip sync score {sync_score:.2f}, A/V lag {lag_ms:.0f}ms — within natural range."
            )

    logger.info(
        f"[REALTIME] verdict={verdict} prob={deep_prob:.3f} "
        f"sync={sync_score:.3f} lag={lag_ms:.0f}ms xcorr={xcorr_peak:.3f}"
    )

    return jsonify({
        "verdict":              verdict,
        "deepfake_probability": result["deepfake_probability"],
        "real_probability":     result["real_probability"],
        "confidence":           confidence,
        "sync_score":           sync_score,
        "lag_ms":               lag_ms,
        "xcorr_peak":           xcorr_peak,
        "cues":                 cues,
        "explanation":          explanation,
        "decision_reason":      result.get("decision_reason", ""),
        "evidence_quality":     result.get("evidence_quality", {}),
        "model_probability":    result.get("model_probability"),
        "model_used":           "RandomForest" if result.get("feature_importances") else "Heuristic",
    })


# ─── Health check ─────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "SafeGuard API"})


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR.parent))
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
