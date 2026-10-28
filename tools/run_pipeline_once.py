"""
Run the real backend pipeline once on a video, for accuracy testing.

Usage: python tools/run_pipeline_once.py <video> [dump.json]
"""
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import _run_pipeline, jobs


def main():
    src = sys.argv[1]
    dump = sys.argv[2] if len(sys.argv) > 2 else None

    job_id = "diag-" + str(int(time.time()))
    work = tempfile.mkdtemp(prefix="dg_run_")
    dst = os.path.join(work, os.path.basename(src))
    shutil.copy2(src, dst)

    jobs[job_id] = {"status": "running", "progress": 0, "stage": "Starting",
                    "result": None, "error": None, "created": time.time()}

    t0 = time.time()
    _run_pipeline(job_id, dst, work)
    elapsed = time.time() - t0

    job = jobs.get(job_id, {})
    if job.get("status") != "done":
        print("FAILED:", job.get("error"))
        sys.stdout.flush()
        os._exit(1)

    r = job["result"]
    meta = r.get("video_meta", {})
    print(f"file={os.path.basename(src)} elapsed={elapsed:.1f}s")
    print(f"verdict={r['verdict']} p_fake={r['deepfake_probability']:.3f} "
          f"confidence={r['confidence']:.3f} reliability={r['reliability_grade']}")
    print(f"analysis: {meta.get('analyzed_frames')} frames @ {meta.get('analysis_fps')} fps, "
          f"duration={meta.get('duration')}s analyzed={meta.get('analyzed_duration')}s "
          f"audio={meta.get('audio_duration')}s")
    print(f"sync={r['sync_score']:.3f} xcorr={r['xcorr_peak']:.3f} "
          f"lag={r['audio_lip_delay_ms']:.0f}ms dtw={r['dtw_distance']:.4f} "
          f"face={r['face_detection_rate']:.2f}")
    print(f"timeline_seconds={len(r.get('timeline', []))} "
          f"suspicious_intervals={len(r.get('suspicious_intervals', []))}")
    print(f"category={r.get('category')}")
    print(f"decision_reason={r.get('decision_reason')}")
    print(f"evidence_quality={r.get('evidence_quality')}")
    print(f"summary={r.get('explanation', {}).get('summary')}")

    if dump:
        with open(dump, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2, default=str)
        print("dumped ->", dump)

    shutil.rmtree(work, ignore_errors=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
