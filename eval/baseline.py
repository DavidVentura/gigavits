"""Baseline: render held-out sentences twice with different noise seeds, score both, and report the
seed spread per voice (the natural variation a new-model difference must exceed to count).

Usage:
    venv/bin/python baseline.py --config configs/laptop.toml --sentences held_out.tsv --voices voices.tsv
        --renderer piper-onnx --bucket <app bucket tts dir> --out <dir> [--seeds 1 2]

sentences.tsv: espeak, idx, text (held-out sentences; each voice renders those of its language)
voices.tsv:    voice, speaker

Writes <out>/seed_<n>/ (WAVs, manifest.tsv, rows.csv, summary.csv) per seed and <out>/spread.csv.
Speaker similarity of every seed is measured against the first seed's renders of the same voice.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

import soundfile as sf

from config import load_config
from manifest import VoiceKey, parse_manifest, write_manifest
from phonemize import phonemize
from render import PiperOnnxRenderer, Renderer, RenderRequest, find_piper_onnx
from scores import MEANS, RATES, ScoreRow, mean_of, pooled_rate, rate
from score import make_context, score_all, write_outputs


@dataclass(frozen=True)
class Sentence:
    espeak: str
    idx: int
    text: str


def read_sentences(path: Path) -> list[Sentence]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        if reader.fieldnames != ["espeak", "idx", "text"]:
            raise ValueError(f"{path}: header {reader.fieldnames}, expected espeak, idx, text")
        sentences = [Sentence(r["espeak"], int(r["idx"]), r["text"]) for r in reader]
    keys = [(s.espeak, s.idx) for s in sentences]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate (espeak, idx)")
    return sentences


def read_voices(path: Path) -> list[VoiceKey]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["voice", "speaker"]:
            raise ValueError(f"{path}: header {reader.fieldnames}, expected voice, speaker")
        return [VoiceKey(r["voice"], r["speaker"]) for r in reader]


def piper_onnx_factory(args: argparse.Namespace) -> Callable[[VoiceKey, int], Renderer]:
    if args.bucket is None:
        raise SystemExit("--renderer piper-onnx needs --bucket")
    return lambda key, seed: PiperOnnxRenderer(find_piper_onnx(args.bucket, key.voice), key, seed)


# The new model registers its renderer here, so its checkpoints render through the same path.
RENDERERS: dict[str, Callable[[argparse.Namespace], Callable[[VoiceKey, int], Renderer]]] = {
    "piper-onnx": piper_onnx_factory,
}

SPREAD_METRICS = tuple(RATES) + MEANS + ("duration_s",)


def _value(rows: list[ScoreRow], metric: str) -> float | None:
    return pooled_rate(rows, metric) if metric in RATES else mean_of(rows, metric)


def _row_value(row: ScoreRow, metric: str) -> float | None:
    return rate(row, metric) if metric in RATES else getattr(row, metric)


def seed_spread(first: list[ScoreRow], second: list[ScoreRow]) -> list[dict[str, object]]:
    """Per voice and metric: both seeds' values and the mean absolute per-sentence difference."""
    def by_key(rows: list[ScoreRow]) -> dict[tuple[str, str, str, int], ScoreRow]:
        return {(r.voice, r.speaker, r.espeak, r.idx): r for r in rows}

    a, b = by_key(first), by_key(second)
    if a.keys() != b.keys():
        raise ValueError(f"seed renders cover different sentences: {sorted(a.keys() ^ b.keys())[:5]}")
    groups: dict[tuple[str, str, str], list[tuple[ScoreRow, ScoreRow]]] = defaultdict(list)
    for key in sorted(a):
        groups[key[:3]].append((a[key], b[key]))
    out = []
    for (voice, speaker, espeak), pairs in groups.items():
        for metric in SPREAD_METRICS:
            value_a = _value([p[0] for p in pairs], metric)
            value_b = _value([p[1] for p in pairs], metric)
            if value_a is None or value_b is None:
                continue
            diffs = [
                abs(x - y)
                for x, y in ((_row_value(p, metric), _row_value(q, metric)) for p, q in pairs)
                if x is not None and y is not None
            ]
            out.append({
                "voice": voice, "speaker": speaker, "espeak": espeak, "metric": metric, "n": len(pairs),
                "seed_a": value_a, "seed_b": value_b, "abs_diff": abs(value_a - value_b),
                "mean_abs_sentence_diff": fmean(diffs) if diffs else None,
            })
    return out


def render_seed(renderers: list[Renderer], sentences: list[Sentence], phonemes: dict[tuple[str, int], str],
                out: Path, reference_set: str, reference_dir: Path | None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for renderer in renderers:
        key = renderer.key
        for s in sentences:
            if s.espeak != renderer.espeak:
                continue
            rendered = renderer.render(RenderRequest(s.espeak, phonemes[(s.espeak, s.idx)]))
            name = f"{key.voice}__{key.speaker}__{s.espeak}__{s.idx}.wav"
            sf.write(out / name, rendered.audio, rendered.sample_rate, subtype="PCM_16")
            row = {
                "voice": key.voice, "speaker": key.speaker, "espeak": s.espeak, "idx": s.idx, "file": name,
                "sample_rate": rendered.sample_rate, "text": s.text, "espeak_phonemes": phonemes[(s.espeak, s.idx)],
                "reference_voice": f"{reference_set}/{key}", "cpu_s": f"{rendered.cpu_s:.4f}",
            }
            if reference_dir is not None:
                row["reference_wav"] = str(reference_dir / name)
            rows.append(row)
    if not rows:
        raise ValueError(f"no sentence matches the voices' languages {sorted({r.espeak for r in renderers})}")
    manifest = out / "manifest.tsv"
    write_manifest(manifest, rows)
    return manifest


def write_spread(path: Path, lines: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(lines[0]))
        writer.writeheader()
        for line in lines:
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else ("" if v is None else v) for k, v in line.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--voices", type=Path, required=True)
    parser.add_argument("--renderer", choices=sorted(RENDERERS), required=True)
    parser.add_argument("--bucket", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=2, default=[1, 2])
    args = parser.parse_args()
    if args.seeds[0] == args.seeds[1]:
        raise SystemExit("the two seeds must differ")

    config = load_config(args.config)
    if config.phonemizer is None:
        raise SystemExit("baseline needs a [phonemizer] to turn sentences into renderer input")
    sentences = read_sentences(args.sentences)
    make_renderer = RENDERERS[args.renderer](args)
    voices = read_voices(args.voices)
    renderers = {seed: [make_renderer(key, seed) for key in voices] for seed in args.seeds}
    languages = {r.espeak for r in renderers[args.seeds[0]]}
    sentences = [s for s in sentences if s.espeak in languages]
    phonemes = dict(zip(((s.espeak, s.idx) for s in sentences), phonemize(config.phonemizer, [(s.espeak, s.text) for s in sentences])))

    first_set = f"seed_{args.seeds[0]}"
    first_dir = args.out / first_set
    manifests = [
        render_seed(renderers[seed], sentences, phonemes, args.out / f"seed_{seed}", first_set,
                    None if seed == args.seeds[0] else first_dir.resolve())
        for seed in args.seeds
    ]
    ctx = make_context(config, {first_set: parse_manifest(manifests[0])})
    scored = []
    for manifest in manifests:
        rows = score_all(parse_manifest(manifest), ctx)
        write_outputs(manifest.parent, rows)
        scored.append(rows)
    write_spread(args.out / "spread.csv", seed_spread(scored[0], scored[1]))
    print(f"{len(voices)} voices x {len(args.seeds)} seeds -> {args.out}")


if __name__ == "__main__":
    main()
