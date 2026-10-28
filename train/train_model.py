"""
train_model.py
Trains a Random Forest classifier using synthetically generated features.
Simulates realistic real vs. deepfake feature distributions.

Run this once to generate the model file:
    python train/train_model.py
"""

import sys
import os
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.pipeline.classifier import CLASSIFIER_FEATURE_KEYS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"
MODEL_DIR.mkdir(exist_ok=True)


def generate_synthetic_features(n_real: int = 600, n_fake: int = 600,
                                  seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic feature vectors for training.

    Real videos:
    - High sync score (0.55–0.90)
    - Low lag (<80ms)
    - High cross-correlation (0.45–0.85)
    - Low DTW distance
    - High alignment score

    Deepfake videos:
    - Low sync score (0.10–0.50)
    - High lag (80–400ms)
    - Low cross-correlation (0.05–0.40)
    - High DTW distance
    - Low alignment score
    """
    rng = np.random.RandomState(seed)
    n_features = len(CLASSIFIER_FEATURE_KEYS)

    X_real = np.zeros((n_real, n_features))
    X_fake = np.zeros((n_fake, n_features))

    # Feature index map
    fmap = {k: i for i, k in enumerate(CLASSIFIER_FEATURE_KEYS)}

    def fill_base(X, n, is_deepfake: bool):
        """Fill baseline feature values with realistic distributions."""
        for k in CLASSIFIER_FEATURE_KEYS:
            i = fmap[k]
            if k.startswith("mfcc_") and "_mean" in k:
                X[:, i] = rng.normal(-20 if not is_deepfake else -18, 15, n)
            elif k.startswith("mfcc_") and "_std" in k:
                X[:, i] = np.abs(rng.normal(8, 3, n))
            elif k.startswith("mel_"):
                X[:, i] = rng.normal(-25, 10, n)
            elif k == "zcr_mean":
                X[:, i] = rng.uniform(0.02, 0.12, n)
            elif k == "rms_mean":
                X[:, i] = rng.uniform(0.01, 0.15, n)
            elif k == "rms_std":
                X[:, i] = rng.uniform(0.005, 0.08, n)
            elif k == "spectral_centroid_mean":
                X[:, i] = rng.uniform(1200, 3500, n)
            elif k == "silence_ratio":
                X[:, i] = rng.beta(2, 5, n)
            elif k == "speech_activity":
                X[:, i] = 1.0 - X[:, fmap["silence_ratio"]]
            elif k == "dynamic_range_db":
                X[:, i] = rng.uniform(10, 40, n)
            else:
                X[:, i] = rng.normal(0.5, 0.15, n)

    fill_base(X_real, n_real, is_deepfake=False)
    fill_base(X_fake, n_fake, is_deepfake=True)

    # ── Override key sync features ──────────────────────────────────────────
    # REAL: high sync, low lag
    X_real[:, fmap["sync_score"]] = np.clip(rng.beta(6, 2, n_real), 0.3, 1.0)
    X_real[:, fmap["xcorr_peak"]] = np.clip(rng.beta(5, 2, n_real), 0.3, 1.0)
    X_real[:, fmap["audio_lip_lag_abs"]] = np.abs(rng.normal(0.02, 0.02, n_real))
    X_real[:, fmap["dtw_distance"]] = np.abs(rng.normal(0.12, 0.06, n_real))
    X_real[:, fmap["dtw_score"]] = np.clip(rng.beta(5, 2, n_real), 0.4, 1.0)
    X_real[:, fmap["cosine_similarity_mean"]] = np.clip(rng.beta(5, 2, n_real), 0.3, 1.0)
    X_real[:, fmap["cosine_similarity_std"]] = np.abs(rng.normal(0.08, 0.03, n_real))
    X_real[:, fmap["alignment_score"]] = np.clip(rng.beta(6, 2, n_real), 0.4, 1.0)
    X_real[:, fmap["suspicious_window_fraction"]] = np.clip(rng.beta(1.5, 6, n_real), 0, 0.5)

    # REAL: natural lip movement
    X_real[:, fmap["face_detection_rate"]] = np.clip(rng.beta(8, 1.5, n_real), 0.5, 1.0)
    X_real[:, fmap["mar_mean"]] = np.clip(rng.normal(0.3, 0.07, n_real), 0.1, 0.7)
    X_real[:, fmap["mar_std"]] = np.abs(rng.normal(0.1, 0.03, n_real))
    X_real[:, fmap["lip_velocity_mean"]] = np.abs(rng.normal(0.04, 0.015, n_real))
    X_real[:, fmap["lip_activity_ratio"]] = np.clip(rng.beta(4, 2, n_real), 0.2, 0.9)

    # DEEPFAKE: low sync, high lag
    X_fake[:, fmap["sync_score"]] = np.clip(rng.beta(2, 6, n_fake), 0.0, 0.65)
    X_fake[:, fmap["xcorr_peak"]] = np.clip(rng.beta(2, 5, n_fake), 0.0, 0.55)
    X_fake[:, fmap["audio_lip_lag_abs"]] = np.abs(rng.normal(0.18, 0.09, n_fake))
    X_fake[:, fmap["dtw_distance"]] = np.abs(rng.normal(0.45, 0.15, n_fake))
    X_fake[:, fmap["dtw_score"]] = np.clip(rng.beta(2, 5, n_fake), 0.0, 0.6)
    X_fake[:, fmap["cosine_similarity_mean"]] = np.clip(rng.beta(2, 5, n_fake), 0.0, 0.55)
    X_fake[:, fmap["cosine_similarity_std"]] = np.abs(rng.normal(0.18, 0.06, n_fake))
    X_fake[:, fmap["alignment_score"]] = np.clip(rng.beta(2, 5, n_fake), 0.0, 0.55)
    X_fake[:, fmap["suspicious_window_fraction"]] = np.clip(rng.beta(4, 2, n_fake), 0.15, 1.0)

    # DEEPFAKE: less natural lip movement
    X_fake[:, fmap["face_detection_rate"]] = np.clip(rng.beta(5, 3, n_fake), 0.3, 1.0)
    X_fake[:, fmap["mar_mean"]] = np.clip(rng.normal(0.25, 0.10, n_fake), 0.05, 0.65)
    X_fake[:, fmap["mar_std"]] = np.abs(rng.normal(0.07, 0.04, n_fake))
    X_fake[:, fmap["lip_velocity_mean"]] = np.abs(rng.normal(0.025, 0.015, n_fake))
    X_fake[:, fmap["lip_activity_ratio"]] = np.clip(rng.beta(2, 4, n_fake), 0.05, 0.7)

    X = np.vstack([X_real, X_fake])
    y = np.array([0] * n_real + [1] * n_fake)  # 0=real, 1=deepfake

    return X, y


def train():
    logger.info("Generating synthetic training data...")
    X, y = generate_synthetic_features(n_real=800, n_fake=800, seed=2024)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    logger.info(f"Features: {len(CLASSIFIER_FEATURE_KEYS)}")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    logger.info("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final training
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Real", "Deepfake"]))
    logger.info(f"Test AUC: {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Feature importance
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    logger.info("\nTop 10 Features by Importance:")
    for rank, idx in enumerate(top_idx):
        logger.info(f"  {rank+1:2d}. {CLASSIFIER_FEATURE_KEYS[idx]:<40s} {importances[idx]:.4f}")

    # Save model
    model_path = MODEL_DIR / "rf_model.pkl"
    joblib.dump(clf, model_path)
    logger.info(f"\nModel saved to {model_path}")

    names_path = MODEL_DIR / "feature_names.pkl"
    joblib.dump(CLASSIFIER_FEATURE_KEYS, names_path)
    logger.info(f"Feature names saved to {names_path}")

    return clf


if __name__ == "__main__":
    train()
