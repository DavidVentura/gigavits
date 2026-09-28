"""Shard manifest records (one JSON object per line of `manifest.jsonl`) and the shared token table."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath


class RecordError(ValueError):
    pass


class Condition(Enum):
    """Recording condition. Declaration order is the condition ID order, so `clean` is always 0."""

    CLEAN = "clean"
    NARROW_BAND = "narrow_band"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class Vocab:
    size: int
    pad: int
    bos: int
    eos: int

    @classmethod
    def load(cls, table_path: Path) -> Vocab:
        raw = json.loads(table_path.read_text(encoding="utf-8"))
        ids = {t["key"]: t["id"] for t in raw["tokens"]}
        if sorted(ids.values()) != list(range(len(ids))):
            raise RecordError(f"{table_path}: token ids must be dense and unique")
        return cls(size=len(ids), pad=ids[raw["pad"]], bos=ids[raw["bos"]], eos=ids[raw["eos"]])


@dataclass(frozen=True)
class Record:
    id: str
    audio: Path
    language: str
    espeak: str
    speaker: str
    condition: Condition
    phoneme_ids: tuple[int, ...]
    durations: tuple[int, ...] | None
    text: str
    teacher: str
    weight: float


_FIELDS = {
    "id", "audio", "language", "espeak", "speaker", "condition",
    "phoneme_ids", "durations", "text", "teacher", "weight",
}


def _string(raw: dict, key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise RecordError(f"{key}: expected a non-empty string, got {value!r}")
    return value


def _int_list(raw: dict, key: str) -> tuple[int, ...]:
    value = raw[key]
    if not isinstance(value, list) or not value:
        raise RecordError(f"{key}: expected a non-empty list of integers")
    if any(type(v) is not int for v in value):
        raise RecordError(f"{key}: expected integers only")
    return tuple(value)


def _audio_path(raw: dict, shard_dir: Path) -> Path:
    relative = PurePosixPath(_string(raw, "audio"))
    if relative.is_absolute() or ".." in relative.parts:
        raise RecordError(f"audio: must be a path inside the shard, got {str(relative)!r}")
    return shard_dir / relative


def _phoneme_ids(raw: dict, vocab: Vocab) -> tuple[int, ...]:
    ids = _int_list(raw, "phoneme_ids")
    if any(not 0 <= i < vocab.size for i in ids):
        raise RecordError(f"phoneme_ids: id outside the {vocab.size}-token table")
    # tokenizer.py emits [bos, pad, t1, pad, ..., tn, pad, eos].
    if len(ids) < 3 or len(ids) % 2 == 0:
        raise RecordError(f"phoneme_ids: length {len(ids)} is not 2n+3")
    if ids[0] != vocab.bos or ids[-1] != vocab.eos:
        raise RecordError("phoneme_ids: must start with BOS and end with EOS")
    if any(i != vocab.pad for i in ids[1:-1:2]):
        raise RecordError("phoneme_ids: PAD must follow BOS and every token")
    return ids


def _durations(raw: dict, length: int) -> tuple[int, ...] | None:
    if raw["durations"] is None:
        return None
    durations = _int_list(raw, "durations")
    if len(durations) != length:
        raise RecordError(f"durations: {len(durations)} values for {length} phoneme ids")
    if any(d < 0 for d in durations) or sum(durations) == 0:
        raise RecordError("durations: must be non-negative with a positive total")
    return durations


def _weight(raw: dict) -> float:
    value = raw["weight"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"weight: expected a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise RecordError(f"weight: must be positive and finite, got {value!r}")
    return float(value)


def parse_record(line: str, shard_dir: Path, vocab: Vocab) -> Record:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as e:
        raise RecordError(f"invalid JSON: {e}") from None
    if not isinstance(raw, dict):
        raise RecordError("expected a JSON object")
    if set(raw) != _FIELDS:
        missing, extra = _FIELDS - set(raw), set(raw) - _FIELDS
        raise RecordError(f"fields: missing {sorted(missing)}, unexpected {sorted(extra)}")
    language = _string(raw, "language")
    # Language IDs key a torch ModuleDict, which forbids dots.
    if "." in language:
        raise RecordError(f"language: {language!r} contains '.'")
    try:
        condition = Condition(raw["condition"])
    except ValueError:
        raise RecordError(f"condition: unknown value {raw['condition']!r}") from None
    if not isinstance(raw["text"], str):
        raise RecordError("text: expected a string")
    phoneme_ids = _phoneme_ids(raw, vocab)
    return Record(
        id=_string(raw, "id"),
        audio=_audio_path(raw, shard_dir),
        language=language,
        espeak=_string(raw, "espeak"),
        speaker=_string(raw, "speaker"),
        condition=condition,
        phoneme_ids=phoneme_ids,
        durations=_durations(raw, len(phoneme_ids)),
        text=raw["text"],
        teacher=_string(raw, "teacher"),
        weight=_weight(raw),
    )


def parse_manifest(lines: list[str], shard_dir: Path, vocab: Vocab) -> list[Record]:
    """Every line must parse; a malformed record aborts with its line number."""
    records = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(parse_record(line, shard_dir, vocab))
        except RecordError as e:
            raise RecordError(f"{shard_dir / 'manifest.jsonl'}:{number}: {e}") from None
    ids = [r.id for r in records]
    if len(set(ids)) != len(ids):
        raise RecordError(f"{shard_dir}: duplicate record ids")
    return records
