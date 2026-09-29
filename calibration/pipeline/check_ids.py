"""Tolerant re-run of verify's token check: per teacher, how many records' phoneme_ids differ from a
fresh espeak phonemization of their text (verify stops at the first). Run from generate/."""
import glob
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from gigagen import TOKENS_DIR
from gigagen.config import load
from gigagen.rt import TeacherRuntime
from tokenizer import Table, piper_input, tokenize

CONFIG = Path(sys.argv[1])


def check(args):
    teacher, espeak, records = args
    cfg = load(CONFIG)
    table = Table.load(TOKENS_DIR / "table.json")
    bad = []
    with TeacherRuntime(cfg.paths.teacher_rt, cfg.paths.espeak_data, {"engine": "espeak", "espeak": espeak}, Path(f"/tmp/check-{teacher}.log")) as rt:
        for rid, text, ids in records:
            fresh = tokenize(piper_input(rt.phonemize(text).phonemes), table.language(espeak), table)
            if tuple(fresh) != tuple(ids):
                bad.append(rid)
    return teacher, len(records), bad


def main():
    by_teacher = defaultdict(list)
    espeak = {}
    for m in glob.glob(sys.argv[2] + "/*/manifest.jsonl"):
        for line in open(m, encoding="utf-8"):
            r = json.loads(line)
            by_teacher[r["teacher"]].append((r["id"], r["text"], r["phoneme_ids"]))
            espeak[r["teacher"]] = r["espeak"]
    with ProcessPoolExecutor(28) as pool:
        results = list(pool.map(check, [(t, espeak[t], rs) for t, rs in by_teacher.items()]))
    total = sum(n for _, n, _ in results)
    bad = sum(len(b) for _, _, b in results)
    for t, n, b in sorted(results):
        if b:
            print(f"{t}\t{len(b)}/{n}\t{' '.join(b[:3])}")
    print(f"{bad} of {total} records differ from a fresh phonemization")


if __name__ == "__main__":
    main()
