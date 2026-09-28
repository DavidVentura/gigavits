"""Score a folder of WAVs described by a manifest (see manifest.py for the columns).

Usage:
    venv/bin/python score.py --config configs/laptop.toml --manifest <manifest.tsv> --out <dir>
        [--reference NAME=<manifest.tsv>]... [--voices voice,voice...]

Writes <dir>/rows.csv (one row per file) and <dir>/summary.csv (one row per voice, speaker and
language). Metrics run only if their config section is present; see config.py.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from align import character_errors
from config import Config, load_config
from f0 import f0_correlation
from languages import Languages, load_languages
from manifest import ReferenceVoice, Utterance, parse_manifest
from models import SAMPLE_RATE, Models
from pairs import PhoneTable, read_pair_counts
from phonemize import phonemize
from phones import Pair, PairStatus, recognized_units, split_by_pair_status, unit_edits
from repo import quality_score
from scores import ScoreRow, summarize, write_rows, write_summary


def load_16k(path: Path, sample_rate: int | None = None) -> np.ndarray:
    """quality/score.py's loader (checks the manifest's sample rate, resamples with soxr_hq)."""
    module = quality_score()
    rate = sf.info(str(path)).samplerate if sample_rate is None else sample_rate
    return module.load_16k(module.Clip(voice="", speaker="", espeak="", idx=0, path=path, sample_rate=rate))


@dataclass
class Context:
    config: Config
    languages: Languages
    models: Models
    table: PhoneTable
    pair_counts: Counter[Pair] | None
    references: dict[str, list[Utterance]]
    _embeddings: dict[ReferenceVoice, np.ndarray] = field(default_factory=dict)

    def reference_embedding(self, ref: ReferenceVoice) -> np.ndarray:
        if ref in self._embeddings:
            return self._embeddings[ref]
        if ref.reference_set not in self.references:
            raise KeyError(f"reference set {ref.reference_set!r} was not given (--reference {ref.reference_set}=...)")
        wanted = self.config.speaker.reference_utterances
        members = sorted((u for u in self.references[ref.reference_set] if u.key == ref.key), key=lambda u: (u.espeak, u.idx))
        if len(members) < wanted:
            raise ValueError(f"{ref}: {len(members)} utterances, speaker similarity needs {wanted}")
        embeddings = np.stack([self.models.speaker(load_16k(u.path, u.sample_rate)) for u in members[:wanted]])
        mean = embeddings.mean(axis=0)
        self._embeddings[ref] = mean / np.linalg.norm(mean)
        return self._embeddings[ref]


def _phoneme_columns(utterance: Utterance, phonemes: str, audio: np.ndarray, ctx: Context) -> dict[str, object]:
    if utterance.espeak in ctx.table.table.unsupported:
        return {}
    words = ctx.table.words_from_espeak(phonemes, utterance.espeak)
    tokens = ctx.models.phonemes(audio)
    reference = [u for w in words for u in w]
    edits = unit_edits(reference, recognized_units(tokens, ctx.table.classes))
    out: dict[str, object] = {"recognized_phones": " ".join(tokens), "per_edits": sum(edits), "per_units": len(reference)}
    if ctx.pair_counts is not None:
        split = split_by_pair_status(words, edits, ctx.pair_counts)
        out |= {
            "per_seen_edits": split[PairStatus.SEEN].edits,
            "per_seen_units": split[PairStatus.SEEN].length,
            "per_unseen_edits": split[PairStatus.UNSEEN].edits,
            "per_unseen_units": split[PairStatus.UNSEEN].length,
        }
    return out


def score_utterance(utterance: Utterance, phonemes: str | None, ctx: Context) -> ScoreRow:
    config = ctx.config
    language = ctx.languages[utterance.espeak]
    audio = load_16k(utterance.path, utterance.sample_rate)
    columns: dict[str, object] = {}

    if config.cer is not None and language.whisper is not None:
        transcript = ctx.models.transcriber(audio, language.whisper)
        errors = character_errors(utterance.text, transcript)
        columns |= {"transcript": transcript, "cer_edits": errors.edits, "cer_chars": errors.length}

    if config.per is not None:
        assert phonemes is not None
        columns |= _phoneme_columns(utterance, phonemes, audio, ctx)

    if config.lid is not None and language.voxlingua is not None:
        posteriors = ctx.models.lid(audio)
        columns |= {"lid_target": posteriors[language.voxlingua], "lid_top": max(posteriors, key=posteriors.get)}

    if config.speaker is not None and utterance.reference_voice is not None:
        reference = ctx.reference_embedding(utterance.reference_voice)
        columns |= {
            "speaker_reference": str(utterance.reference_voice),
            "speaker_similarity": float(np.dot(ctx.models.speaker(audio), reference)),
        }

    if config.f0 is not None and utterance.reference_wav is not None:
        result = f0_correlation(audio, load_16k(utterance.reference_wav), SAMPLE_RATE, config.f0)
        columns |= {"f0_corr": result.correlation, "f0_status": result.status.value}

    if config.quality is not None:
        columns |= ctx.models.quality(audio)

    return ScoreRow(
        voice=utterance.key.voice,
        speaker=utterance.key.speaker,
        espeak=utterance.espeak,
        idx=utterance.idx,
        text=utterance.text,
        path=str(utterance.path.resolve()),
        duration_s=len(audio) / SAMPLE_RATE,
        cpu_s=utterance.cpu_s,
        **columns,
    )


def phonemes_for(utterances: list[Utterance], config: Config) -> list[str | None]:
    """The espeak string of every utterance, from the manifest or else the configured phonemizer."""
    if config.per is None:
        return [None] * len(utterances)
    missing = [i for i, u in enumerate(utterances) if u.espeak_phonemes is None]
    if missing and config.phonemizer is None:
        raise ValueError(f"{len(missing)} utterances have no espeak_phonemes and no [phonemizer] is configured")
    made = phonemize(config.phonemizer, [(utterances[i].espeak, utterances[i].text) for i in missing]) if missing else []
    out: list[str | None] = [u.espeak_phonemes for u in utterances]
    for i, phonemes in zip(missing, made):
        out[i] = phonemes
    return out


def make_context(config: Config, references: dict[str, list[Utterance]]) -> Context:
    torch.set_num_threads(config.threads)
    pair_counts = None
    if config.per is not None and config.per.pair_counts is not None:
        pair_counts = read_pair_counts(config.per.pair_counts)
    return Context(config, load_languages(), Models(config), PhoneTable.load(), pair_counts, references)


def score_all(utterances: list[Utterance], ctx: Context) -> list[ScoreRow]:
    phonemes = phonemes_for(utterances, ctx.config)
    rows = []
    for n, (utterance, p) in enumerate(zip(utterances, phonemes), 1):
        rows.append(score_utterance(utterance, p, ctx))
        print(f"\r{n}/{len(utterances)} {utterance.path.name}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return rows


def write_outputs(out: Path, rows: list[ScoreRow]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "rows.csv", rows)
    write_summary(out / "summary.csv", summarize(rows))


def parse_reference(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or "/" in name:
        raise argparse.ArgumentTypeError(f"--reference {text!r} is not NAME=<manifest.tsv>")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=parse_reference, action="append", default=[])
    parser.add_argument("--voices", help="comma-separated voice names to score (default: all)")
    args = parser.parse_args()

    utterances = parse_manifest(args.manifest)
    if args.voices:
        wanted = set(args.voices.split(","))
        unknown = wanted - {u.key.voice for u in utterances}
        if unknown:
            sys.exit(f"voices not in the manifest: {sorted(unknown)}")
        utterances = [u for u in utterances if u.key.voice in wanted]
    names = [name for name, _ in args.reference]
    if len(set(names)) != len(names):
        sys.exit("--reference names must be unique")
    references = {name: parse_manifest(path) for name, path in args.reference}
    ctx = make_context(load_config(args.config), references)
    rows = score_all(utterances, ctx)
    write_outputs(args.out, rows)
    print(f"{len(rows)} files -> {args.out}/rows.csv, summary.csv")


if __name__ == "__main__":
    main()
