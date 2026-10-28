import sys
sys.path.insert(0, '.')
from train_lavdf import extract_features_for_video, is_fake, load_metadata
from pathlib import Path

m = load_metadata()
reals = [v for v in m if v['split']=='train' and not is_fake(v)][:1]
fakes = [v for v in m if v['split']=='train' and is_fake(v)][:1]

for v in reals+fakes:
    p = Path(r'E:\LAV-DF\LAV-DF') / v['file']
    print('Testing', p, flush=True)
    try:
        feats = extract_features_for_video(p)
        print('  OK, keys:', len(feats), 'fake=', is_fake(v), flush=True)
        # show a few key features
        for k in ['sync_score','xcorr_peak','audio_lip_lag_abs','dtw_distance','face_detection_rate']:
            print(f'    {k}={feats.get(k)}', flush=True)
    except Exception as e:
        print('  ERR:', e, flush=True)
        import traceback
        traceback.print_exc()
        break
