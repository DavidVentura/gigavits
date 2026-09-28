"""Listening material for slow links: Opus clips and the Phase 1.3 listening page.

Usage:
    venv/bin/python listening.py opus <manifest.tsv> <out dir> [--voices v,v] [--bitrate 24k]
        Converts a manifest's clips to Opus and writes <out dir>/manifest.tsv pointing at them
        (e.g. the verdict manifests from criteria.py, before fetching them to the laptop).
    venv/bin/python listening.py page --out <dir> [--ab ab.tsv] [--rating rating.tsv] [--seed N]
        Builds <dir>/index.html and <dir>/audio/*.opus. Answers stay in the browser (per listener)
        and export as ab.csv / ratings.csv for criteria.py.

ab.tsv:     voice, speaker, language, item, text, original, candidate   (WAV paths; >= 30 items per voice)
rating.tsv: language, voice, speaker, item, text, file                  (10 items per language from 2-3 non-native voices)
Relative paths are relative to the TSV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

from manifest import VoiceKey, parse_manifest, write_manifest
from repo import EVAL

TEMPLATE = EVAL / "listening.template.html"


def to_opus(wav: Path, out: Path, bitrate: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav), "-ac", "1", "-c:a", "libopus",
         "-b:a", bitrate, "-application", "audio", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {wav}: {result.stderr}")


def opus_name(wav: Path) -> str:
    # Clip names repeat across folders (seed_1/x.wav, seed_2/x.wav), so the name carries a path hash.
    digest = hashlib.sha1(str(wav.resolve()).encode()).hexdigest()[:8]
    return f"{wav.stem}-{digest}.opus"


@dataclass(frozen=True)
class AbItem:
    key: VoiceKey
    language: str
    item: str
    text: str
    original: Path
    candidate: Path


@dataclass(frozen=True)
class RatingItem:
    language: str
    key: VoiceKey
    item: str
    text: str
    file: Path


def _read_tsv(path: Path, header: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        if reader.fieldnames != header:
            raise ValueError(f"{path}: header {reader.fieldnames}, expected {header}")
        return list(reader)


def read_ab(path: Path) -> list[AbItem]:
    rows = _read_tsv(path, ["voice", "speaker", "language", "item", "text", "original", "candidate"])
    items = [AbItem(VoiceKey(r["voice"], r["speaker"]), r["language"], r["item"], r["text"],
                    path.parent / r["original"], path.parent / r["candidate"]) for r in rows]
    ids = [(i.key, i.item) for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate (voice, speaker, item)")
    return items


def read_rating(path: Path) -> list[RatingItem]:
    rows = _read_tsv(path, ["language", "voice", "speaker", "item", "text", "file"])
    items = [RatingItem(r["language"], VoiceKey(r["voice"], r["speaker"]), r["item"], r["text"], path.parent / r["file"]) for r in rows]
    ids = [(i.language, i.key, i.item) for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate (language, voice, speaker, item)")
    return items


def ab_groups(items: list[AbItem], rng: random.Random, audio: dict[Path, str]) -> list[dict]:
    """One group per lineup voice; the original's side and the sentence order are drawn at random."""
    by_voice: dict[tuple[VoiceKey, str], list[AbItem]] = {}
    for i in items:
        by_voice.setdefault((i.key, i.language), []).append(i)
    groups = []
    for (key, language), members in sorted(by_voice.items()):
        trials = []
        for m in rng.sample(members, len(members)):
            original_side = rng.choice("AB")
            a, b = (m.original, m.candidate) if original_side == "A" else (m.candidate, m.original)
            trials.append({"id": f"ab|{key}|{language}|{m.item}", "item": m.item, "text": m.text,
                           "a": audio[a], "b": audio[b], "original_side": original_side})
        groups.append({"label": f"{key.voice} ({language}, {len(trials)} sentences)", "voice": key.voice,
                       "speaker": key.speaker, "language": language, "items": trials})
    return groups


def rating_groups(items: list[RatingItem], rng: random.Random, audio: dict[Path, str]) -> list[dict]:
    by_language: dict[str, list[RatingItem]] = {}
    for i in items:
        by_language.setdefault(i.language, []).append(i)
    return [
        {"label": f"{language} ({len(members)} clips)", "language": language,
         "items": [{"id": f"rating|{language}|{m.key}|{m.item}", "item": m.item, "voice": m.key.voice,
                    "speaker": m.key.speaker, "text": m.text, "audio": audio[m.file]}
                   for m in rng.sample(members, len(members))]}
        for language, members in sorted(by_language.items())
    ]


def build_page(out: Path, ab: list[AbItem], rating: list[RatingItem], seed: int, bitrate: str) -> None:
    wavs = sorted({p for i in ab for p in (i.original, i.candidate)} | {i.file for i in rating})
    audio = {}
    for wav in wavs:
        name = f"audio/{opus_name(wav)}"
        to_opus(wav, out / name, bitrate)
        audio[wav] = name
    rng = random.Random(seed)
    data = {"ab": ab_groups(ab, rng, audio), "rating": rating_groups(rating, rng, audio)}
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/{}", json.dumps(data, ensure_ascii=False))
    (out / "index.html").write_text(page, encoding="utf-8")
    size = sum(p.stat().st_size for p in (out / "audio").iterdir())
    print(f"{len(ab)} A/B trials, {len(rating)} rating clips, {len(wavs)} Opus files ({size / 1e6:.1f} MB) -> {out}/index.html")


def convert_manifest(manifest: Path, out: Path, voices: set[str] | None, bitrate: str) -> None:
    utterances = parse_manifest(manifest)
    if voices is not None:
        utterances = [u for u in utterances if u.key.voice in voices]
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = {(r["voice"], r["speaker"], r["espeak"], r["idx"]): r
                for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)}
    out_rows = []
    for u in utterances:
        name = opus_name(u.path)
        to_opus(u.path, out / name, bitrate)
        row = dict(rows[(u.key.voice, u.key.speaker, u.espeak, str(u.idx))])
        row["file"] = name
        out_rows.append(row)
    write_manifest(out / "manifest.tsv", out_rows)
    print(f"{len(out_rows)} clips -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    opus = sub.add_parser("opus")
    opus.add_argument("manifest", type=Path)
    opus.add_argument("out", type=Path)
    opus.add_argument("--voices")
    opus.add_argument("--bitrate", default="24k")
    page = sub.add_parser("page")
    page.add_argument("--out", type=Path, required=True)
    page.add_argument("--ab", type=Path)
    page.add_argument("--rating", type=Path)
    page.add_argument("--seed", type=int, default=1)
    page.add_argument("--bitrate", default="24k")
    args = parser.parse_args()

    if args.command == "opus":
        convert_manifest(args.manifest, args.out, None if args.voices is None else set(args.voices.split(",")), args.bitrate)
        return
    if args.ab is None and args.rating is None:
        raise SystemExit("page needs --ab and/or --rating")
    ab = [] if args.ab is None else read_ab(args.ab)
    rating = [] if args.rating is None else read_rating(args.rating)
    build_page(args.out, ab, rating, args.seed, args.bitrate)


if __name__ == "__main__":
    main()
