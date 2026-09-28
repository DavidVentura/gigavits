import numpy as np, soundfile as sf, os, unicodedata

def alpha_count(text):
    return sum(1 for ch in text if unicodedata.category(ch).startswith('L'))

def save(outdir, voice, idx, audio, sr, text):
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1)
    fn = f"{voice}__0__{idx}.wav"
    sf.write(os.path.join(outdir, fn), np.clip(a, -1, 1), int(sr), subtype='PCM_16')
    return fn, len(a) / sr

def write_manifest(outdir, espeak, rows):
    with open(os.path.join(outdir, 'manifest.tsv'), 'w') as f:
        f.write('voice\tspeaker\tespeak\tidx\tfile\tsample_rate\tphonemes\ttext\n')
        for voice, idx, fn, sr, text in rows:
            t = ' '.join(text.split())
            f.write(f"{voice}\t0\t{espeak}\t{idx}\t{fn}\t{int(sr)}\t{alpha_count(t)}\t{t}\n")
