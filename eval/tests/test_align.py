from align import EditCount, Op, align, character_errors, edit_count, normalize_text


def test_identical_sequences_match_everywhere():
    steps = align("abc", "abc")
    assert [s.op for s in steps] == [Op.MATCH] * 3


def test_substitution_deletion_insertion():
    assert edit_count("kitten", "sitting") == EditCount(3, 6)
    ops = [s.op for s in align("abc", "ac")]
    assert ops == [Op.MATCH, Op.DELETE, Op.MATCH]
    ops = [s.op for s in align("ac", "abc")]
    assert ops == [Op.MATCH, Op.INSERT, Op.MATCH]


def test_alignment_indices_cover_both_sequences():
    steps = align("abcdef", "azcdxef")
    assert [s.ref for s in steps if s.ref is not None] == list(range(6))
    assert [s.hyp for s in steps if s.hyp is not None] == list(range(7))


def test_empty_sides():
    assert edit_count("", "ab") == EditCount(2, 0)
    assert edit_count("ab", "") == EditCount(2, 2)


def test_normalize_text_drops_case_punctuation_and_extra_space():
    assert normalize_text('  Hola, ¿qué   tal?  "Bien".') == "hola qué tal bien"


def test_character_errors_ignore_punctuation_and_case():
    assert character_errors("Das ist ein Test.", "das ist ein test") == EditCount(0, 16)
    assert character_errors("Das ist ein Test.", "das ist kein test").edits == 1


def test_rate_of_empty_reference_raises():
    import pytest

    with pytest.raises(ZeroDivisionError):
        EditCount(1, 0).rate
