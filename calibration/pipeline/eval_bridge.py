"""Turns a `gigatrain render` output (renders.tsv) into eval/score.py manifests.

    python eval_bridge.py <render_dir> <shard_dir>... [--reference-count 10]

Writes <render_dir>/student.tsv (the renders, each with reference_voice teacher/<voice>|<speaker>)
and <render_dir>/teacher.tsv (the teacher's audio of the same ids plus reference utterances per
speaker), so both can be scored with the same metrics and compared.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

SAMPLE_RATE = 22050
HEAD = ["voice", "speaker", "espeak", "idx", "file", "sample_rate", "text", "reference_voice"]


def split_key(speaker_key: str) -> tuple[str, str]:
    voice, sep, speaker = speaker_key.partition(".")
    return voice, speaker if sep else "0"


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEAD, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in HEAD})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("render_dir", type=Path)
    p.add_argument("shards", type=Path, nargs="+")
    p.add_argument("--reference-count", type=int, default=10)
    args = p.parse_args()

    records = {}
    by_speaker = defaultdict(list)
    for shard in args.shards:
        for line in (shard / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            r["path"] = str(shard / r["audio"])
            records[r["id"]] = r
            by_speaker[r["speaker"]].append(r)

    with (args.render_dir / "renders.tsv").open(encoding="utf-8") as f:
        renders = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

    student, teacher = [], []
    rendered_ids = {x["id"] for x in renders}
    wanted = sorted({x["speaker"] for x in renders})
    # reference utterances take the lowest idx: score.py averages the first N by idx
    idx = defaultdict(int)
    for key in wanted:
        voice, speaker = split_key(key)
        others = sorted((r for r in by_speaker[key] if r["id"] not in rendered_ids), key=lambda r: r["id"])
        for r in others[: args.reference_count]:
            n = idx[key]
            idx[key] += 1
            teacher.append({"voice": voice, "speaker": speaker, "espeak": r["espeak"], "idx": n, "file": r["path"],
                            "sample_rate": SAMPLE_RATE, "text": r["text"]})
    for x in renders:
        r = records[x["id"]]
        voice, speaker = split_key(x["speaker"])
        n = idx[x["speaker"]]
        idx[x["speaker"]] += 1
        common = {"voice": voice, "speaker": speaker, "espeak": r["espeak"], "idx": n, "sample_rate": SAMPLE_RATE,
                  "text": r["text"], "reference_voice": f"teacher/{voice}|{speaker}"}
        student.append(common | {"file": str(Path(x["path"]).resolve())})
        teacher.append(common | {"file": r["path"]})
    write(args.render_dir / "student.tsv", student)
    write(args.render_dir / "teacher.tsv", teacher)
    print(f"{len(student)} student rows, {len(teacher)} teacher rows over {len(wanted)} speakers")


if __name__ == "__main__":
    main()
