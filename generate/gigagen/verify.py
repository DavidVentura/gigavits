"""Checks packaged shards the way the training loader will read them."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import soundfile
from tokenizer import Table, piper_input, tokenize

from . import TOKENS_DIR
from .align import HOP, TARGET_RATE
from .catalog import Condition
from .config import Config
from .rt import TeacherRuntime


class ManifestError(ValueError):
    pass


FIELDS = {
    "id": str, "audio": str, "language": str, "espeak": str, "speaker": str, "condition": str,
    "phoneme_ids": list, "durations": (list, type(None)), "text": str, "teacher": str, "weight": float,
}


@dataclass(frozen=True)
class Record:
    id: str
    audio: str
    language: str
    espeak: str
    speaker: str
    condition: Condition
    phoneme_ids: tuple[int, ...]
    durations: tuple[int, ...] | None
    text: str
    teacher: str
    weight: float


def parse_record(line: str) -> Record:
    raw = json.loads(line)
    if set(raw) != set(FIELDS):
        raise ManifestError(f"fields {sorted(raw)} != {sorted(FIELDS)}")
    for key, kind in FIELDS.items():
        if not isinstance(raw[key], kind):
            raise ManifestError(f"{raw['id']}: {key} is {type(raw[key]).__name__}")
    ids = tuple(raw["phoneme_ids"])
    durations = None if raw["durations"] is None else tuple(raw["durations"])
    if not all(isinstance(x, int) for x in ids) or (durations is not None and not all(isinstance(x, int) and x >= 0 for x in durations)):
        raise ManifestError(f"{raw['id']}: phoneme_ids and durations must be non-negative integers")
    if durations is not None and len(durations) != len(ids):
        raise ManifestError(f"{raw['id']}: {len(durations)} durations for {len(ids)} phoneme IDs")
    return Record(**{**raw, "condition": Condition(raw["condition"]), "phoneme_ids": ids, "durations": durations})


def verify(config: Config) -> None:
    table = Table.load(TOKENS_DIR / "table.json")
    shards = sorted((config.paths.out / "shards").glob("shard-*"))
    if not shards:
        raise ManifestError("no shards")
    records: list[tuple[Path, Record]] = []
    for shard in shards:
        for line in (shard / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            records.append((shard, parse_record(line)))
    ids = [r.id for _, r in records]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate item IDs")
    seconds: Counter[str] = Counter()
    by_espeak: dict[str, list[Record]] = {}
    for shard, r in records:
        info = soundfile.info(shard / r.audio)
        if (info.samplerate, info.channels, info.format) != (TARGET_RATE, 1, "FLAC"):
            raise ManifestError(f"{r.id}: {info.samplerate} Hz, {info.channels} channels, {info.format}")
        audio, _ = soundfile.read(shard / r.audio, dtype="float32")
        if r.durations is not None and sum(r.durations) * HOP != audio.size:
            raise ManifestError(f"{r.id}: sum(durations) * {HOP} = {sum(r.durations) * HOP} != {audio.size}")
        seconds[r.teacher] += audio.size / TARGET_RATE
        by_espeak.setdefault(r.espeak, []).append(r)
    log = config.paths.out / "logs" / "verify-teacher-rt.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for espeak, rs in sorted(by_espeak.items()):
        language = table.language(espeak)
        with TeacherRuntime(config.paths.teacher_rt, config.paths.espeak_data, {"engine": "espeak", "espeak": espeak}, log) as rt:
            for r in rs:
                if tuple(tokenize(piper_input(rt.phonemize(r.text).phonemes), language, table)) != r.phoneme_ids:
                    raise ManifestError(f"{r.id}: phoneme_ids differ from the text's espeak tokens")
    with_durations = sum(1 for _, r in records if r.durations is not None)
    print(f"verified {len(records)} items in {len(shards)} shards ({with_durations} with durations)")
    for teacher, s in sorted(seconds.items()):
        print(f"  {teacher}\t{s:.1f} s")
