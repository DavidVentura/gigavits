"""Checkpoint evaluation renders: held-out sentences in a few speakers per language, plus the
eval/score.py manifests (student.tsv with speaker references, teacher.tsv with each speaker's
reference utterances from the shards).

    train/.venv/bin/python render_eval.py --config C --run-dir R --weights W --held-out H.jsonl \
        --out DIR [--speakers-per-language 2] [--sentences 5] [--device cuda]

Run from train/. Renders at Piper's defaults (noise 0.667, length 1.0, noise_w 0.8), clean condition.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch

from gigatrain.__main__ import _model_state, _vocab
from gigatrain.config import load_config, to_jsonable
from gigatrain.ids import IdMaps
from gigatrain.records import Condition
from gigatrain.training import BackEndJob

HEAD = ["voice", "speaker", "espeak", "idx", "file", "sample_rate", "text", "espeak_phonemes", "reference_voice"]
REFERENCE_UTTERANCES = 10


def split_key(key: str) -> tuple[str, str]:
    voice, sep, speaker = key.partition(".")
    return voice, speaker if sep else "0"


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEAD, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in HEAD})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--held-out", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--speakers-per-language", type=int, default=2)
    p.add_argument("--sentences", type=int, default=5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    config = load_config(args.config)
    ids = IdMaps.load(args.run_dir / "ids.json")
    job = BackEndJob(to_jsonable(config), ids.to_json(), _vocab(config).size)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)["state_dict"]
    job.model.load_state_dict(_model_state(state))
    model = job.model.eval().to(args.device)

    held_out = defaultdict(list)
    for line in args.held_out.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        held_out[r["language"]].append(r)

    # teacher references straight from the shards the run trains on
    by_speaker = defaultdict(list)
    for shard in config.data.shards:
        for line in (Path(shard) / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            by_speaker[r["speaker"]].append((r["id"], str(Path(shard) / r["audio"]), r["espeak"], r["text"]))

    student, teacher = [], []
    cid = torch.tensor([ids.cid(Condition.CLEAN)], device=args.device)
    for language in ids.languages:
        speakers = sorted(ids.speakers_of(language))[: args.speakers_per_language]
        lid = torch.tensor([ids.lid(language)], device=args.device)
        (args.out / language).mkdir(parents=True, exist_ok=True)
        for key in speakers:
            voice, speaker = split_key(key)
            refs = sorted(by_speaker[key])[:REFERENCE_UTTERANCES]
            for n, (_, path, espeak, text) in enumerate(refs):
                teacher.append({"voice": voice, "speaker": speaker, "espeak": espeak, "idx": n, "file": path,
                                "sample_rate": config.audio.sample_rate, "text": text})
            sid = torch.tensor([ids.sid(key)], device=args.device)
            for n, r in enumerate(held_out[language][: args.sentences]):
                x = torch.tensor([r["phoneme_ids"]], device=args.device)
                with torch.no_grad():
                    audio = model.infer(x, torch.tensor([x.size(1)], device=args.device), sid, lid, cid, 0.667, 1.0, 0.8, model.dec)
                path = args.out / language / f"{key}-{n}.wav"
                sf.write(path, audio.squeeze().float().cpu().numpy(), config.audio.sample_rate)
                student.append({"voice": voice, "speaker": speaker, "espeak": r["espeak"], "idx": n, "file": str(path.resolve()),
                                "sample_rate": config.audio.sample_rate, "text": r["text"], "espeak_phonemes": r["espeak_phonemes"],
                                "reference_voice": f"teacher/{voice}|{speaker}"})
    write(args.out / "student.tsv", student)
    write(args.out / "teacher.tsv", teacher)
    print(f"{len(student)} renders, {len(teacher)} teacher references -> {args.out}")


if __name__ == "__main__":
    main()
