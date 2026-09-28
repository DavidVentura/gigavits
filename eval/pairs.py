"""Phoneme-pair counts of the training data, for the unseen-pair split of phoneme error.

Usage:
    venv/bin/python pairs.py <out pairs.tsv> <shard manifest.jsonl>...

Reads training shard manifests (one JSON object per line: id, audio, language, espeak, speaker,
condition, phoneme_ids, durations, text, teacher, weight) and counts adjacent within-word phone
pairs of the shared-table phoneme IDs, the same units the scorer builds from espeak strings.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from phones import Pair, Word, count_pairs, reference_words
from repo import TOKENS, tokens_module

HEADER = ("first", "second", "count")


@dataclass(frozen=True)
class PhoneTable:
    """The shared token table seen as phone units."""

    table: object
    key_by_id: dict[int, str]
    classes: dict[str, str]

    @classmethod
    def load(cls) -> PhoneTable:
        tokenizer = tokens_module("tokenizer")
        table = tokenizer.Table.load(TOKENS / "table.json")
        return cls(table, {v: k for k, v in table.ids.items()}, table.classes)

    def words_from_espeak(self, phonemes: str, espeak: str) -> list[Word]:
        tokenizer = tokens_module("tokenizer")
        ids = tokenizer.symbol_ids(phonemes, self.table.language(espeak), self.table)
        return reference_words([self.key_by_id[i] for i in ids], self.classes)

    def words_from_model_ids(self, phoneme_ids: Iterable[int]) -> list[Word]:
        specials = {self.table.pad, self.table.bos, self.table.eos}
        keys = []
        for i in phoneme_ids:
            if i not in self.key_by_id:
                raise ValueError(f"phoneme id {i} is not in the shared table")
            if i not in specials:
                keys.append(self.key_by_id[i])
        return reference_words(keys, self.classes)


@dataclass(frozen=True)
class ShardItem:
    id: str
    audio: str
    language: str
    espeak: str
    speaker: str
    condition: str
    phoneme_ids: tuple[int, ...]
    durations: tuple[int, ...] | None
    text: str
    teacher: str
    weight: float


class ShardFormatError(ValueError):
    pass


def _typed(record: dict, key: str, kind: type | tuple[type, ...]):
    if key not in record:
        raise ShardFormatError(f"missing {key!r}")
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, kind):
        raise ShardFormatError(f"{key!r} is {type(value).__name__}, expected {kind}")
    return value


def _int_list(record: dict, key: str) -> tuple[int, ...]:
    values = _typed(record, key, list)
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        raise ShardFormatError(f"{key!r} holds non-integers")
    return tuple(values)


def parse_shard_line(line: str) -> ShardItem:
    record = json.loads(line)
    if not isinstance(record, dict):
        raise ShardFormatError("line is not a JSON object")
    durations = None if record.get("durations", ...) is None else _int_list(record, "durations")
    phoneme_ids = _int_list(record, "phoneme_ids")
    if durations is not None and len(durations) != len(phoneme_ids):
        raise ShardFormatError(f"{len(durations)} durations for {len(phoneme_ids)} phoneme ids")
    return ShardItem(
        id=_typed(record, "id", str),
        audio=_typed(record, "audio", str),
        language=_typed(record, "language", str),
        espeak=_typed(record, "espeak", str),
        speaker=_typed(record, "speaker", str),
        condition=_typed(record, "condition", str),
        phoneme_ids=phoneme_ids,
        durations=durations,
        text=_typed(record, "text", str),
        teacher=_typed(record, "teacher", str),
        weight=float(_typed(record, "weight", (int, float))),
    )


def read_shard_manifest(path: Path) -> Iterator[ShardItem]:
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield parse_shard_line(line)
            except (ShardFormatError, json.JSONDecodeError) as e:
                raise ShardFormatError(f"{path}:{n}: {e}") from e


def training_pair_counts(items: Iterable[ShardItem], table: PhoneTable) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for item in items:
        counts.update(count_pairs(table.words_from_model_ids(item.phoneme_ids)))
    return counts


def write_pair_counts(path: Path, counts: Counter[Pair]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(HEADER) + "\n")
        for (a, b), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{a}\t{b}\t{n}\n")


def read_pair_counts(path: Path) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    with path.open(encoding="utf-8") as f:
        header = tuple(f.readline().rstrip("\n").split("\t"))
        if header != HEADER:
            raise ValueError(f"{path}: header {header}, expected {HEADER}")
        for n, line in enumerate(f, 2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{n}: expected 3 fields, got {fields}")
            pair = (fields[0], fields[1])
            if pair in counts:
                raise ValueError(f"{path}:{n}: duplicate pair {pair}")
            counts[pair] = int(fields[2])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("manifests", type=Path, nargs="+")
    args = parser.parse_args()
    table = PhoneTable.load()
    counts: Counter[Pair] = Counter()
    for manifest in args.manifests:
        counts += training_pair_counts(read_shard_manifest(manifest), table)
    write_pair_counts(args.out, counts)
    print(f"{len(counts)} pair types, {sum(counts.values())} pairs from {len(args.manifests)} manifests -> {args.out}")


if __name__ == "__main__":
    main()
