import json

with open(r'E:\LAV-DF\LAV-DF\metadata.json') as f:
    m = json.load(f)

print(type(m), len(m))
for i in range(4):
    print('\n===', i, '===')
    print(json.dumps(m[i], indent=2))
