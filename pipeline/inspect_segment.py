import json

with open('backend/jobs/ac3d037fc573/transcript.json', 'r', encoding='utf-8') as f:
    tr = json.load(f)

for s in tr.get('segments', []):
    if s['start'] >= 75 and s['start'] <= 80:
        print('SEGMENT START:', s['start'], 'END:', s['end'])
        print('TEXT:', s['text'])
        print('--- WORDS ---')
        for w in s['words']:
            print(f"{w['start']:.2f} - {w['end']:.2f}: {repr(w['word'])}")

