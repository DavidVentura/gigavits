import json, collections, requests, io, sys, time, soundfile as sf
sys.path.insert(0, '.')
from save_util import save

def pick(meta, n_spk=3):
    by = collections.defaultdict(list)
    for x in meta:
        by[x['speaker_id']].append(x)
    cands = []
    for sp, xs in by.items():
        g = sorted([x for x in xs if 4 <= x['duration'] <= 10], key=lambda x: -x['c50'])
        if len(g) >= 3:
            cands.append((sum(x['c50'] for x in g[:3]) / 3, sp, xs[0]['gender'], g[:3]))
    cands.sort(reverse=True)
    chosen, genders = [], collections.Counter()
    for c in cands:
        if genders[c[2]] >= 2:
            continue
        chosen.append(c)
        genders[c[2]] += 1
        if len(chosen) == n_spk:
            break
    return chosen

def fetch_row(ds, idx):
    for attempt in range(8):
        r = requests.get(f"https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split=train&offset={idx}&length=1")
        if r.status_code == 200:
            return r.json()['rows'][0]['row']
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"rows api failed {r.status_code}")

ds, metafile, outdir, lang = sys.argv[1:5]
meta = json.load(open(metafile))
rows = []
for c50, sp, gender, utts in pick(meta):
    voice = f"ivr-{lang}{sp}"
    for k, u in enumerate(utts):
        row = fetch_row(ds, u['row_idx'])
        assert row['speaker_id'] == sp and abs(row['duration'] - u['duration']) < 1e-6
        b = requests.get(row['audio'][0]['src']).content
        a, sr = sf.read(io.BytesIO(b))
        fn, dur = save(outdir, voice, k, a, sr, row['text'])
        print(voice, 'gender', gender, 'c50', round(u['c50'], 1), 'snr', round(u['snr'], 1), sr, round(dur, 2), row['samples'], row['text'], flush=True)
        rows.append((voice, k, fn, sr, row['text']))
        time.sleep(2)
json.dump(rows, open(f'{lang}_ivr_rows.json', 'w'), ensure_ascii=False)
