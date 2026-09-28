import pytest

from scores import ScoreRow, pooled_rate, read_rows, summarize, write_rows


def row(idx=0, **kw):
    base = dict(voice="v", speaker="0", espeak="de", idx=idx, text=f"t{idx}", path=f"/a{idx}.wav", duration_s=2.0)
    return ScoreRow(**(base | kw))


def test_roundtrip_keeps_types_and_empties(tmp_path):
    rows = [row(0, cer_edits=3, cer_chars=30, UTMOS=3.5, transcript='a, "b"'), row(1, lid_top="de", f0_status="ok")]
    path = tmp_path / "rows.csv"
    write_rows(path, rows)
    back = read_rows(path)
    assert back[0].cer_edits == 3 and back[0].UTMOS == pytest.approx(3.5) and back[0].transcript == 'a, "b"'
    assert back[1].cer_edits is None and back[1].lid_top == "de"


def test_rates_pool_numerators_and_denominators():
    rows = [row(0, per_edits=1, per_units=10), row(1, per_edits=9, per_units=30), row(2)]
    assert pooled_rate(rows, "per") == pytest.approx(10 / 40)
    assert pooled_rate([row(0)], "per") is None


def test_summary_groups_by_voice_and_language():
    rows = [row(0, UTMOS=3.0, cpu_s=0.2), row(1, UTMOS=4.0, cpu_s=0.2), row(0, espeak="it", UTMOS=2.0)]
    lines = {l["espeak"]: l for l in summarize(rows)}
    assert lines["de"]["n"] == 2 and lines["de"]["UTMOS"] == pytest.approx(3.5) and lines["de"]["UTMOS_min"] == 3.0
    assert lines["de"]["cpu_s_per_audio_s"] == pytest.approx(0.1)
    assert lines["it"]["cpu_s_per_audio_s"] is None and lines["it"]["cer"] is None


def test_missing_required_value_fails(tmp_path):
    path = tmp_path / "rows.csv"
    write_rows(path, [row(0)])
    text = path.read_text().replace(",2,", ",,", 1)
    path.write_text(text)
    with pytest.raises(ValueError):
        read_rows(path)
