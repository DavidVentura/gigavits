import json
from collections import Counter

import pytest

from pairs import (
    PhoneTable,
    ShardFormatError,
    parse_shard_line,
    read_pair_counts,
    read_shard_manifest,
    training_pair_counts,
    write_pair_counts,
)
from phones import count_pairs
from repo import tokens_module

TABLE = PhoneTable.load()
tokenizer = tokens_module("tokenizer")


def item(phonemes: str, language: str, **overrides) -> dict:
    ids = tokenizer.tokenize(phonemes, TABLE.table.language(language), TABLE.table)
    record = {
        "id": "x", "audio": "a.flac", "language": language, "espeak": language, "speaker": "s", "condition": "clean",
        "phoneme_ids": ids, "durations": [1] * len(ids), "text": "t", "teacher": "piper", "weight": 1.0,
    }
    return record | overrides


def test_model_ids_give_the_same_pairs_as_the_espeak_string():
    phonemes = "tˈuːtːɪ kˌome stˈate? "
    from_ids = count_pairs(TABLE.words_from_model_ids(parse_shard_line(json.dumps(item(phonemes, "it"))).phoneme_ids))
    from_string = count_pairs(TABLE.words_from_espeak(phonemes, "it"))
    assert from_ids == from_string
    assert from_ids[("t", "u")] == 1 and from_ids[("u", "t")] == 1


def test_counts_accumulate_over_items(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(item("das ɪst. ", "de")) for _ in range(3)) + "\n", encoding="utf-8")
    counts = training_pair_counts(read_shard_manifest(manifest), TABLE)
    assert counts[("d", "a")] == 3 and counts[("s", "t")] == 3
    assert ("s", "ɪ") not in counts


def test_null_durations_are_allowed():
    assert parse_shard_line(json.dumps(item("das. ", "de", durations=None))).durations is None


@pytest.mark.parametrize("change", [
    {"phoneme_ids": "1 2 3"},
    {"phoneme_ids": [1, 2.5]},
    {"weight": "high"},
    {"durations": [1]},
    {"speaker": 3},
])
def test_malformed_items_fail(change):
    with pytest.raises(ShardFormatError):
        parse_shard_line(json.dumps(item("das. ", "de") | change))


def test_missing_key_fails():
    record = item("das. ", "de")
    del record["durations"]
    with pytest.raises(ShardFormatError):
        parse_shard_line(json.dumps(record))


def test_unknown_ids_fail():
    with pytest.raises(ValueError):
        TABLE.words_from_model_ids([1, 9999])


def test_error_names_the_line(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(item("das. ", "de")) + "\n{\"id\": 1}\n", encoding="utf-8")
    with pytest.raises(ShardFormatError, match="manifest.jsonl:2"):
        list(read_shard_manifest(manifest))


def test_pair_count_file_roundtrip(tmp_path):
    counts = Counter({("a", "b"): 3, ("ʃ", "iː"): 1})
    path = tmp_path / "pairs.tsv"
    write_pair_counts(path, counts)
    assert read_pair_counts(path) == counts
