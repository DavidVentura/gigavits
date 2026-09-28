from collections import Counter

import pytest

from align import edit_count
from pairs import PhoneTable
from phones import (
    MalformedPhonemes,
    PairStatus,
    Unit,
    count_pairs,
    recognized_units,
    reference_words,
    split_by_pair_status,
    unit_edits,
    unit_pair_status,
)

TABLE = PhoneTable.load()


def labels(words):
    return [[u.label for u in w] for w in words]


def test_stress_is_dropped_and_marks_attach():
    words = TABLE.words_from_espeak("tˈuːtːɪ kˌome. ", "it")
    assert labels(words) == [["t", "uː", "tː", "ɪ"], ["k", "o", "m", "e"]]


def test_punctuation_and_space_break_words():
    words = TABLE.words_from_espeak("das ɪst, aɪn tˈɛst. ", "de")
    assert labels(words) == [["d", "a", "s"], ["ɪ", "s", "t"], ["a", "ɪ", "n"], ["t", "ɛ", "s", "t"]]


def test_context_rules_apply_before_units():
    # hi: '.' after r is the retroflex diacritic, not punctuation, so it stays inside the word.
    words = TABLE.words_from_espeak("pər.aː kə. ", "hi")
    assert labels(words) == [["p", "ə", "r̢", "aː"], ["k", "ə"]]


def test_mark_without_phone_raises_on_reference():
    with pytest.raises(MalformedPhonemes):
        reference_words(["ː", "a"], TABLE.classes)


def test_recognized_units_split_multichar_tokens_and_drop_notation():
    units = recognized_units(["aɪ", "tʃ", "iː", "s.", "i5", "ʲ"], TABLE.classes)
    assert [u.label for u in units] == ["a", "ɪ", "t", "ʃ", "iː", "s", "iʲ"]


def test_leading_mark_in_recognizer_output_is_ignored():
    assert [u.label for u in recognized_units(["ː", "a"], TABLE.classes)] == ["a"]


def test_marks_are_order_insensitive():
    assert reference_words(["t", "ʰ", "ː"], TABLE.classes) == reference_words(["t", "ː", "ʰ"], TABLE.classes)


def units(text):
    return [Unit(c, "") for c in text]


def test_unit_edits_sum_to_edit_distance():
    for ref, hyp in [("abcd", "abcd"), ("abcd", "xabd"), ("abcd", "abcdxx"), ("abcd", ""), ("ab", "xxab")]:
        edits = unit_edits(units(ref), units(hyp))
        assert sum(edits) == edit_count(ref, hyp).edits, (ref, hyp, edits)
        assert len(edits) == len(ref)


def test_insertions_are_charged_to_the_preceding_unit():
    assert unit_edits(units("abc"), units("abxc")) == [0, 1, 0]
    assert unit_edits(units("abc"), units("xabc")) == [1, 0, 0]


def test_empty_reference_raises():
    with pytest.raises(MalformedPhonemes):
        unit_edits([], units("a"))


def test_pairs_are_within_words_on_base_phones():
    words = [(Unit("a", "ː"), Unit("b", ""), Unit("c", "")), (Unit("d", ""),)]
    assert count_pairs(words) == Counter({("a", "b"): 1, ("b", "c"): 1})


def test_pair_status_and_split():
    words = [(Unit("a", ""), Unit("b", ""), Unit("c", "")), (Unit("d", ""),)]
    trained = Counter({("a", "b"): 5})
    statuses = unit_pair_status(words, trained)
    assert statuses == [PairStatus.SEEN, PairStatus.UNSEEN, PairStatus.UNSEEN, PairStatus.SEEN]
    split = split_by_pair_status(words, [1, 0, 2, 1], trained)
    assert (split[PairStatus.SEEN].edits, split[PairStatus.SEEN].length) == (2, 2)
    assert (split[PairStatus.UNSEEN].edits, split[PairStatus.UNSEEN].length) == (2, 2)


def test_split_rejects_mismatched_edits():
    with pytest.raises(ValueError):
        split_by_pair_status([(Unit("a", ""),)], [0, 0], Counter())
