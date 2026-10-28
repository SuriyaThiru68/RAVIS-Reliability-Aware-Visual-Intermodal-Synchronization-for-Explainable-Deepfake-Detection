"""
run.py — SafeGuard launcher
Run this from the project root:
    python run.py
"""
import sys
import os

# Ensure root is in path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

if __name__ == "__main__":
    print("=" * 55)
    print("  SafeGuard — Deepfake Detection System")
    print("=" * 55)

    # Check if model exists; train if not
    model_path = os.path.join(ROOT, "backend", "models", "rf_model.pkl")
    if not os.path.exists(model_path):
        print("\n[INFO] No trained model found. Training now...")
        from train.train_model import train
        train()
        print("[INFO] Model trained and saved.\n")
    else:
        print("[INFO] Pre-trained model found.")

    print("\n[INFO] Starting Flask server on http://localhost:5000")
    print("[INFO] Open your browser at: http://localhost:5000\n")
    print("Press CTRL+C to stop.\n")

    from backend.app import app
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
