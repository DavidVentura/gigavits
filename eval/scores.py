"""Typed per-file score rows, their CSV form, and the per-voice summary.

An empty cell means the metric was not configured or does not apply to the file (e.g. CER in a
language outside the Whisper allow-list, F0 when durations do not align). Rates are stored as
their numerator and denominator so that summaries pool them exactly; the rate columns written
next to them are for reading only and are ignored when a CSV is parsed back.
"""
from __future__ import annotations

import csv
import types
import typing
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import fmean
from typing import Iterable

from manifest import VoiceKey


@dataclass(frozen=True)
class ScoreRow:
    voice: str
    speaker: str
    espeak: str
    idx: int
    text: str
    path: str
    duration_s: float
    cpu_s: float | None = None
    transcript: str | None = None
    cer_edits: int | None = None
    cer_chars: int | None = None
    recognized_phones: str | None = None
    per_edits: int | None = None
    per_units: int | None = None
    per_seen_edits: int | None = None
    per_seen_units: int | None = None
    per_unseen_edits: int | None = None
    per_unseen_units: int | None = None
    lid_target: float | None = None
    lid_top: str | None = None
    speaker_reference: str | None = None
    speaker_similarity: float | None = None
    f0_corr: float | None = None
    f0_status: str | None = None
    UTMOS: float | None = None
    SIG: float | None = None
    BAK: float | None = None
    OVRL: float | None = None
    P808: float | None = None

    @property
    def key(self) -> VoiceKey:
        return VoiceKey(self.voice, self.speaker)


RATES = {
    "cer": ("cer_edits", "cer_chars"),
    "per": ("per_edits", "per_units"),
    "per_seen": ("per_seen_edits", "per_seen_units"),
    "per_unseen": ("per_unseen_edits", "per_unseen_units"),
}
MEANS = ("lid_target", "speaker_similarity", "f0_corr", "UTMOS", "SIG", "BAK", "OVRL", "P808")


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def rate(row: ScoreRow, name: str) -> float | None:
    numerator, denominator = RATES[name]
    return _ratio(getattr(row, numerator), getattr(row, denominator))


_TYPES = typing.get_type_hints(ScoreRow)


def _parse_cell(name: str, cell: str) -> object:
    hint = _TYPES[name]
    optional = isinstance(hint, types.UnionType) and type(None) in hint.__args__
    base = next(a for a in hint.__args__ if a is not type(None)) if optional else hint
    if cell == "":
        if optional:
            return None
        if base is str:
            return ""
        raise ValueError(f"{name} is empty")
    return base(cell)


def _format(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_rows(path: Path, rows: Iterable[ScoreRow]) -> None:
    names = [f.name for f in fields(ScoreRow)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(names + list(RATES))
        for row in rows:
            writer.writerow([_format(getattr(row, n)) for n in names] + [_format(rate(row, r)) for r in RATES])


def read_rows(path: Path) -> list[ScoreRow]:
    names = [f.name for f in fields(ScoreRow)]
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        absent = [n for n in names if n not in (reader.fieldnames or ())]
        if absent:
            raise ValueError(f"{path}: no column {absent}")
        out = []
        for n, record in enumerate(reader, 2):
            try:
                out.append(ScoreRow(**{name: _parse_cell(name, record[name]) for name in names}))
            except ValueError as e:
                raise ValueError(f"{path}:{n}: {e}") from e
    return out


def pooled_rate(rows: Iterable[ScoreRow], name: str) -> float | None:
    numerator, denominator = RATES[name]
    pairs = [(getattr(r, numerator), getattr(r, denominator)) for r in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return None
    return _ratio(sum(a for a, _ in pairs), sum(b for _, b in pairs))


def mean_of(rows: Iterable[ScoreRow], name: str) -> float | None:
    values = [getattr(r, name) for r in rows if getattr(r, name) is not None]
    return fmean(values) if values else None


def summarize(rows: Iterable[ScoreRow]) -> list[dict[str, object]]:
    """One line per (voice, speaker, language)."""
    groups: dict[tuple[str, str, str], list[ScoreRow]] = defaultdict(list)
    for r in rows:
        groups[(r.voice, r.speaker, r.espeak)].append(r)
    out = []
    for (voice, speaker, espeak), members in sorted(groups.items()):
        line: dict[str, object] = {"voice": voice, "speaker": speaker, "espeak": espeak, "n": len(members)}
        for name in RATES:
            line[name] = pooled_rate(members, name)
        for name in MEANS:
            line[name] = mean_of(members, name)
        utmos = [r.UTMOS for r in members if r.UTMOS is not None]
        line["UTMOS_min"] = min(utmos) if utmos else None
        timed = [r for r in members if r.cpu_s is not None]
        line["cpu_s_per_audio_s"] = (
            sum(r.cpu_s for r in timed) / sum(r.duration_s for r in timed) if timed else None
        )
        out.append(line)
    return out


def write_summary(path: Path, lines: list[dict[str, object]]) -> None:
    if not lines:
        raise ValueError("nothing to summarize")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(lines[0]))
        writer.writeheader()
        writer.writerows({k: _format(v) for k, v in line.items()} for line in lines)
