"""Scoring manifests: the voices/samples/manifest.tsv format plus optional evaluation columns.

Required: voice, speaker, espeak, idx, file, sample_rate, text.
Optional:
    espeak_phonemes   the piper-rs espeak string of `text` (NFD); otherwise the configured phonemizer makes it
    reference_wav     the same sentence by the reference system, for F0 contour correlation
    reference_voice   `<reference set>/<voice>|<speaker>`: the voice whose mean embedding speaker
                      similarity is measured against, looked up in the reference manifests given to score.py
    cpu_s             CPU seconds spent rendering the file (runtime hook of render.py)
Other columns (e.g. the samples' phoneme count) are ignored. Paths are relative to the manifest.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REQUIRED = ("voice", "speaker", "espeak", "idx", "file", "sample_rate", "text")
OPTIONAL = ("espeak_phonemes", "reference_wav", "reference_voice", "cpu_s")


class ManifestError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class VoiceKey:
    voice: str
    speaker: str

    def __str__(self) -> str:
        return f"{self.voice}|{self.speaker}"

    @classmethod
    def parse(cls, text: str) -> VoiceKey:
        voice, sep, speaker = text.partition("|")
        if not sep or not voice or not speaker or "|" in speaker:
            raise ManifestError(f"voice key {text!r} is not '<voice>|<speaker>'")
        return cls(voice, speaker)


@dataclass(frozen=True, order=True)
class ReferenceVoice:
    reference_set: str
    key: VoiceKey

    def __str__(self) -> str:
        return f"{self.reference_set}/{self.key}"

    @classmethod
    def parse(cls, text: str) -> ReferenceVoice:
        reference_set, sep, rest = text.partition("/")
        if not sep or not reference_set:
            raise ManifestError(f"reference voice {text!r} is not '<reference set>/<voice>|<speaker>'")
        return cls(reference_set, VoiceKey.parse(rest))


@dataclass(frozen=True)
class Utterance:
    key: VoiceKey
    espeak: str
    idx: int
    path: Path
    sample_rate: int
    text: str
    espeak_phonemes: str | None
    reference_wav: Path | None
    reference_voice: ReferenceVoice | None
    cpu_s: float | None

    @property
    def id(self) -> str:
        return f"{self.key}|{self.espeak}|{self.idx}"


def _optional(row: dict[str, str], column: str) -> str | None:
    value = row.get(column)
    return value if value else None


def parse_row(row: dict[str, str], base: Path) -> Utterance:
    missing = [c for c in REQUIRED if not row.get(c)]
    if missing:
        raise ManifestError(f"missing {missing}")
    reference_wav = _optional(row, "reference_wav")
    reference_voice = _optional(row, "reference_voice")
    cpu_s = _optional(row, "cpu_s")
    return Utterance(
        key=VoiceKey(row["voice"], row["speaker"]),
        espeak=row["espeak"],
        idx=int(row["idx"]),
        path=base / row["file"],
        sample_rate=int(row["sample_rate"]),
        text=row["text"],
        espeak_phonemes=_optional(row, "espeak_phonemes"),
        reference_wav=None if reference_wav is None else base / reference_wav,
        reference_voice=None if reference_voice is None else ReferenceVoice.parse(reference_voice),
        cpu_s=None if cpu_s is None else float(cpu_s),
    )


def parse_manifest(path: Path) -> list[Utterance]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        absent = [c for c in REQUIRED if c not in (reader.fieldnames or ())]
        if absent:
            raise ManifestError(f"{path}: no column {absent}")
        utterances = []
        for n, row in enumerate(reader, 2):
            try:
                utterances.append(parse_row(row, path.parent))
            except (ManifestError, ValueError) as e:
                raise ManifestError(f"{path}:{n}: {e}") from e
    duplicates = sorted(i for i, n in Counter(u.id for u in utterances).items() if n > 1)
    if duplicates:
        raise ManifestError(f"{path}: duplicate utterances {duplicates[:5]}")
    return utterances


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    # Column order of voices/samples/manifest.tsv, whose phoneme count is carried when present.
    head = ["voice", "speaker", "espeak", "idx", "file", "sample_rate"]
    counted = ["phonemes"] if any("phonemes" in r for r in rows) else []
    columns = head + counted + ["text"] + [c for c in OPTIONAL if any(c in r for r in rows)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writeheader()
        for r in rows:
            for c in ("text", "espeak_phonemes"):
                if "\t" in str(r.get(c, "")) or "\n" in str(r.get(c, "")):
                    raise ManifestError(f"{c} of {r['file']} contains a tab or newline")
            writer.writerow({c: r.get(c, "") for c in columns})
