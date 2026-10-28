#!/usr/bin/env python3
"""
Train RandomForest on LAV-DF dataset using the project's exact feature extraction pipeline.
"""
import json
import os
import sys
import random
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from backend.pipeline.classifier import CLASSIFIER_FEATURE_KEYS, build_feature_vector
from backend.pipeline.video_processor import validate_video, extract_frames, extract_audio
from backend.pipeline.face_detector import FaceDetector
from backend.pipeline.audio_features import extract_all_audio_features
from backend.pipeline.visual_features import extract_visual_features, extract_lip_signal
from backend.pipeline.sync_analyzer import compute_sync_features, resample_to_common_grid
from backend.pipeline.emotion_analyzer import run_emotion_analysis


LAVDF_ROOT = Path(r"E:\LAV-DF\LAV-DF")
METADATA = LAVDF_ROOT / "metadata.json"
MODEL_DIR = Path(__file__).parent / "backend" / "models"


def load_metadata():
    with open(METADATA, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_fake(v):
    if v.get('n_fakes', 0) > 0:
        return True
    if v.get('modify_video') or v.get('modify_audio'):
        return True
    if v.get('original') is not None:
        return True
    return False


def extract_features_for_video(video_path):
    work_dir = tempfile.mkdtemp(prefix="lavdf_train_")
    try:
        meta = validate_video(str(video_path))
        if not meta.get("valid"):
            raise ValueError(meta.get("error"))
        
        # Extract audio
        audio_path = extract_audio(str(video_path), work_dir)
        
        # Extract frames
        frames, original_fps, sampling_fps = extract_frames(
            str(video_path), max_frames=800, duration_hint=meta["duration"]
        )
        
        # Face detection
        detector = FaceDetector()
        face_results = detector.process_frames(frames)
        detector.close()
        
        # Audio features
        audio_feats = extract_all_audio_features(audio_path) if audio_path else {}
        whisper_info = audio_feats.pop("_whisper", {"available": False})
        
        # Visual features
        visual_feats = extract_visual_features(face_results, fps=sampling_fps)
        lip_signal = extract_lip_signal(face_results)
        import numpy as _np_lv
        lip_velocity_signal = _np_lv.array(
            [r.lip_velocity for r in face_results if r.detected],
            dtype=_np_lv.float64,
        ) if face_results else _np_lv.array([], dtype=_np_lv.float64)
        
        # Sync
        energy_envelope = audio_feats.pop("_energy_envelope", np.zeros(0))
        audio_events = audio_feats.pop("_audio_events", [])
        audio_duration = float(audio_feats.get("_audio_duration", 0.0) or 0.0)
        
        analysis_duration = float(meta.get("duration", 0.0) or 0.0)
        lip_duration = len(lip_signal) / sampling_fps if sampling_fps > 0 else 0.0
        if lip_duration > 0:
            analysis_duration = min(analysis_duration, lip_duration) if analysis_duration > 0 else lip_duration
        if energy_envelope is not None and len(energy_envelope) > 0:
            analysis_duration = min(analysis_duration, len(energy_envelope) / 25.0)
        
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
            whisper_info=whisper_info,
        )
        
        # Emotion
        emotion_analysis = run_emotion_analysis(frames, face_results, audio_feats)
        
        # Combine
        combined = {}
        combined.update({k: v for k, v in audio_feats.items() if not k.startswith("_")})
        combined.update({k: v for k, v in visual_feats.items() if not k.startswith("_")})
        combined.update({k: v for k, v in sync_feats.items() if not k.startswith("_")})
        combined["_active_second_count"] = sync_feats.get("_active_second_count", 0)
        combined["_video_duration"] = meta.get("duration", 0.0)
        combined["_analysis_duration"] = analysis_duration
        combined["emotion_disagreement"] = float(emotion_analysis.get("emotion_disagreement", 0.0) or 0.0)
        
        return combined
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    print("Loading metadata...")
    meta = load_metadata()
    
    # Build label map
    labels = {}
    for v in meta:
        fpath = v['file']  # like test/000001.mp4
        labels[fpath] = 1 if is_fake(v) else 0
    
    # Get training videos
    train_videos = [v for v in meta if v.get('split') == 'train']
    real_train = [v for v in train_videos if labels[v['file']] == 0]
    fake_train = [v for v in train_videos if labels[v['file']] == 1]
    
    print(f"Train split: REAL={len(real_train)}, FAKE={len(fake_train)}")
    
    # Sample subset for faster training
    random.seed(42)
    sample_size = 800  # 800 each = 1600 videos, should be enough
    real_sample = random.sample(real_train, min(sample_size, len(real_train)))
    fake_sample = random.sample(fake_train, min(sample_size, len(fake_train)))
    
    samples = real_sample + fake_sample
    random.shuffle(samples)
    
    print(f"Sampling {len(samples)} videos for training...")
    
    X_list = []
    y_list = []
    failed = 0
    
    for i, v in enumerate(samples):
        rel_path = v['file']
        full_path = LAVDF_ROOT / rel_path
        try:
            print(f"[{i+1}/{len(samples)}] Processing {rel_path}...", flush=True)
            feats = extract_features_for_video(full_path)
            vec = build_feature_vector(feats)
            X_list.append(vec)
            y_list.append(labels[rel_path])
            print(f"  OK - label={labels[rel_path]}", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            failed += 1
            if failed > 50:
                print("Too many failures, stopping")
                break
    
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"\nExtracted {len(X)} samples, {failed} failed")
    print(f"Class distribution: REAL={(y==0).sum()}, FAKE={(y==1).sum()}")
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    
    print("\nRunning cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    print("Training...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    print("\n" + classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print(f"Test AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Feature importance
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\nTop 15 Features:")
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1:2d}. {CLASSIFIER_FEATURE_KEYS[idx]:<40s} {importances[idx]:.4f}")
    
    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "rf_model.pkl"
    names_path = MODEL_DIR / "feature_names.pkl"
    joblib.dump(clf, model_path)
    joblib.dump(CLASSIFIER_FEATURE_KEYS, names_path)
    print(f"\nSaved model to {model_path}")
    print(f"Saved feature names to {names_path}")


if __name__ == '__main__':
    main()
