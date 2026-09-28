import random
from pathlib import Path

import pytest

from baseline import seed_spread
from listening import AbItem, ab_groups, opus_name
from manifest import VoiceKey
from scores import ScoreRow


def row(idx, utmos, per_edits):
    return ScoreRow(voice="v", speaker="0", espeak="de", idx=idx, text=str(idx), path=f"/{idx}", duration_s=2.0,
                    UTMOS=utmos, per_edits=per_edits, per_units=10)


def test_seed_spread_pairs_sentences():
    a = [row(0, 3.0, 1), row(1, 3.4, 3)]
    b = [row(0, 3.2, 2), row(1, 3.0, 3)]
    spread = {line["metric"]: line for line in seed_spread(a, b)}
    assert spread["UTMOS"]["seed_a"] == pytest.approx(3.2) and spread["UTMOS"]["seed_b"] == pytest.approx(3.1)
    assert spread["UTMOS"]["mean_abs_sentence_diff"] == pytest.approx(0.3)
    assert spread["per"]["seed_a"] == pytest.approx(0.2) and spread["per"]["mean_abs_sentence_diff"] == pytest.approx(0.05)
    assert "cer" not in spread


def test_seed_spread_needs_the_same_sentences():
    with pytest.raises(ValueError):
        seed_spread([row(0, 3.0, 1)], [row(1, 3.0, 1)])


def test_ab_sides_are_random_but_recorded():
    key = VoiceKey("v", "0")
    items = [AbItem(key, "de", str(i), f"t{i}", Path(f"/o{i}.wav"), Path(f"/c{i}.wav")) for i in range(40)]
    audio = {p: p.stem for i in items for p in (i.original, i.candidate)}
    [group] = ab_groups(items, random.Random(1), audio)
    assert len(group["items"]) == 40
    sides = {t["original_side"] for t in group["items"]}
    assert sides == {"A", "B"}
    for t in group["items"]:
        original = t["a"] if t["original_side"] == "A" else t["b"]
        assert original.startswith("o") and original[1:] == t["item"]
    assert [t["item"] for t in group["items"]] != [str(i) for i in range(40)]


def test_opus_names_differ_per_folder():
    assert opus_name(Path("/a/x.wav")) != opus_name(Path("/b/x.wav"))
