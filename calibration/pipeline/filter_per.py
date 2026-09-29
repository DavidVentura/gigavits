"""Drops manifest rows whose espeak phonemes eval's phone parser rejects (e.g. a Tamil word that
espeak starts with a bare palatalization mark), so phoneme error can run on the rest.

    eval/venv/bin/python filter_per.py <manifest.tsv>     (run from eval/; rewrites in place, keeps <manifest>.all)
"""
import csv
import shutil
import sys
from pathlib import Path

from pairs import PhoneTable
from phones import MalformedPhonemes

path = Path(sys.argv[1])
shutil.copy(path, path.with_suffix(".all.tsv"))
with path.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
    fields, rows = reader.fieldnames, list(reader)
table = PhoneTable.load()
kept, dropped = [], []
for r in rows:
    try:
        if r["espeak"] not in table.table.unsupported:
            table.words_from_espeak(r["espeak_phonemes"], r["espeak"])
        kept.append(r)
    except MalformedPhonemes:
        dropped.append(r)
with path.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
    w.writeheader()
    w.writerows(kept)
print(f"kept {len(kept)}, dropped {len(dropped)}: " + " ".join(f"{r['voice']}#{r['idx']}" for r in dropped))
