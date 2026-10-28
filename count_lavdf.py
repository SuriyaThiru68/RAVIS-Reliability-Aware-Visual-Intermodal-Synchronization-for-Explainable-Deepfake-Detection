import json
from collections import Counter
with open(r'E:\LAV-DF\LAV-DF\metadata.json') as f:
    m = json.load(f)
c = Counter()
for v in m:
    if v.get('n_fakes',0)>0 or v.get('modify_video') or v.get('modify_audio') or v.get('original') is not None:
        c['FAKE'] +=1
    else:
        c['REAL'] +=1
print('Total:', len(m))
print(c)
c2 = Counter()
for v in m:
    s = v.get('split','?')
    if v.get('n_fakes',0)>0 or v.get('modify_video') or v.get('modify_audio') or v.get('original') is not None:
        c2[(s,'FAKE')]+=1
    else:
        c2[(s,'REAL')]+=1
for k in ['train','dev','test']:
    print(k, c2[(k,'REAL')], c2[(k,'FAKE')])
