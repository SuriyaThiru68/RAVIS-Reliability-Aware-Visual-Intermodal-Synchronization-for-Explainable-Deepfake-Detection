# RAVIS Detection Pipeline
# Stages implemented per the RAVIS paper:
#   1. Preprocessing         — video_processor.py
#   2. Face/Lip Kinematics   — face_detector.py
#   3. Acoustic Extraction   — audio_features.py  (+ Whisper, SNR)
#   4. Synchronization       — sync_analyzer.py   (+ FastDTW, Wasserstein)
#   5. Emotion Consistency   — emotion_analyzer.py
#   6. Reliability Estimation— explainer.py       (SNR + emotion weighted)
#   7. Classification        — classifier.py      (+ SHAP, conformal prediction)
#   8. Reporting             — report_pdf.py      (SHA-256, thumbnails)
