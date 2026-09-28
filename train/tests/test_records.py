import json
from pathlib import Path

import pytest

from conftest import TOKEN_TABLE
from gigatrain.ids import IdError, IdMaps, build_id_maps, require_known
from gigatrain.records import Condition, RecordError, Vocab, parse_manifest, parse_record

VOCAB = Vocab.load(TOKEN_TABLE)
SHARD = Path("/shard")


def raw(**overrides) -> dict:
    record = {
        "id": "a",
        "audio": "audio/a.flac",
        "language": "es-419",
        "espeak": "es-419",
        "speaker": "es_MX-ald",
        "condition": "clean",
        "phoneme_ids": [VOCAB.bos, VOCAB.pad, 20, VOCAB.pad, 21, VOCAB.pad, VOCAB.eos],
        "durations": [1, 2, 3, 4, 5, 6, 7],
        "text": "hola",
        "teacher": "piper:es_MX-ald-medium",
        "weight": 1.5,
    }
    return record | overrides


def parse(**overrides):
    return parse_record(json.dumps(raw(**overrides)), SHARD, VOCAB)


def test_vocab_is_the_shared_table():
    assert (VOCAB.size, VOCAB.pad, VOCAB.bos, VOCAB.eos) == (175, 0, 1, 2)


def test_valid_record():
    r = parse()
    assert r.audio == SHARD / "audio" / "a.flac"
    assert r.condition is Condition.CLEAN
    assert r.durations == (1, 2, 3, 4, 5, 6, 7)
    assert r.weight == 1.5


def test_aligned_record_has_no_durations():
    assert parse(durations=None).durations is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"condition": "noisy"},
        {"weight": 0},
        {"weight": True},
        {"weight": float("inf")},
        {"audio": "/abs/a.flac"},
        {"audio": "../a.flac"},
        {"language": "en.us"},
        {"speaker": ""},
        {"durations": [1, 2, 3]},
        {"durations": [1, 2, -3, 4, 5, 6, 7]},
        {"durations": [0] * 7},
        {"phoneme_ids": [VOCAB.bos, VOCAB.pad, 20, VOCAB.pad, VOCAB.eos, 3, 4]},
        {"phoneme_ids": [VOCAB.bos, 20, 21, VOCAB.pad, VOCAB.eos]},
        {"phoneme_ids": [VOCAB.bos, VOCAB.pad, 175, VOCAB.pad, VOCAB.eos]},
        {"phoneme_ids": [VOCAB.bos, VOCAB.pad, True, VOCAB.pad, VOCAB.eos]},
        {"phoneme_ids": [VOCAB.pad, VOCAB.pad, 20, VOCAB.pad, VOCAB.eos]},
        {"text": None},
    ],
)
def test_malformed_record_raises(overrides):
    with pytest.raises(RecordError):
        parse(**overrides)


def test_missing_and_extra_fields_raise():
    record = raw()
    del record["teacher"]
    with pytest.raises(RecordError, match="missing"):
        parse_record(json.dumps(record), SHARD, VOCAB)
    with pytest.raises(RecordError, match="unexpected"):
        parse_record(json.dumps(raw(extra=1)), SHARD, VOCAB)


def test_manifest_reports_line_and_rejects_duplicates():
    good = json.dumps(raw())
    with pytest.raises(RecordError, match=":2:"):
        parse_manifest([good, "{"], SHARD, VOCAB)
    with pytest.raises(RecordError, match="duplicate"):
        parse_manifest([good, good], SHARD, VOCAB)


def records(*spec):
    return [parse(id=f"{i}", language=lang, espeak=lang, speaker=spk) for i, (lang, spk) in enumerate(spec)]


def test_id_maps_are_append_only():
    first = build_id_maps(records(("es", "a"), ("en-us", "b"), ("en-us", "c")))
    assert first.languages == ("es", "en-us") and first.speakers == ("a", "b", "c")
    second = build_id_maps(records(("pl", "d"), ("en-us", "a")), previous=first)
    assert second.languages == ("es", "en-us", "pl")
    assert second.speakers == ("a", "b", "c", "d")
    assert second.speakers_of("en-us") == ("a", "b", "c")
    assert IdMaps.from_json(json.loads(json.dumps(second.to_json()))) == second
    assert second.cid(Condition.CLEAN) == 0


def test_language_with_two_espeak_voices_raises():
    rs = [parse(id="1", language="es-AR", espeak="es-419"), parse(id="2", language="es-AR", espeak="es")]
    with pytest.raises(IdError):
        build_id_maps(rs)


def test_unknown_speaker_language_pair_raises():
    ids = build_id_maps(records(("es", "a")))
    with pytest.raises(IdError):
        require_known(records(("en-us", "a")), ids)


def test_conditions_known_to_the_code_are_appended_to_an_older_map():
    older = IdMaps(("a",), ("es",), (("es", "es"),), (("es", "a"),), (Condition.CLEAN, Condition.DEGRADED))
    grown = build_id_maps(records(("es", "a")), previous=older)
    assert grown.conditions == (Condition.CLEAN, Condition.DEGRADED, Condition.NARROW_BAND)
    assert grown.cid(Condition.NARROW_BAND) == 2
