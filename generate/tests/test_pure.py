from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from gigagen.align import AlignmentError, rescale_durations
from gigagen.audio import fit_frames
from gigagen.catalog import (
    CatalogError, Condition, CorpusSource, KokoroJaSource, Mimic3Source, MmsSource, PhonemeType, PiperSource, PiperVoiceConfig,
    Speaker, Tier, build_speakers, language_id, parse_other_list, parse_regions, parse_tiers,
)
from gigagen.config import Hours, RatePolicy
from gigagen.coverage import Candidate, greedy_order, held_out
from gigagen.package import Item, assign_shards, teacher_reports
from gigagen.plan import hours_for, jobs_for, part_utterances, tempo_factors, utterance_order
from gigagen.quality import TeacherTiming, duration_outliers, longest_internal_silence
from gigagen.render import JobStats
from gigagen.sentences import Phonemized, select
from gigagen.textclass import Limits, Reject, filter_texts, reject_reason

LIMITS = Limits(4, 25)
REGIONS = parse_regions(
    "# comment\nlocale\tlanguage\nes_ES\tes\nes_MX\tes-419\nes_AR\tes-AR\nen_GB\ten-gb\n"
)


# --- text class filter ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,reason",
    [
        ("The cat sat on the mat.", None),
        ("Too short here.", Reject.TOO_SHORT),
        (" ".join(["word"] * 26), Reject.TOO_LONG),
        ("He bought 3 apples at the market.", Reject.DIGIT),
        ("Il a acheté ٣ pommes au marché.", Reject.DIGIT),
        ("See www.example.org for all the details.", Reject.URL),
        ("It costs five € at the shop today.", Reject.SYMBOL),
        ("Rain fell on 50% of the fields.", Reject.DIGIT),
        ("Rain fell on half & half of the fields.", Reject.SYMBOL),
        ("The U.S. army arrived there in spring.", Reject.ABBREVIATION),
        ("We met Dr. Smith at the station yesterday.", Reject.ABBREVIATION),
        ("They brought apples, pears, etc. and many more.", Reject.ABBREVIATION),
        ("The NASA team launched the rocket today.", Reject.ABBREVIATION),
        ("He arrived in Rome. Then he left again quickly.", None),
        ("Москва является столицей России, сказал Putin вчера.", Reject.MIXED_SCRIPT),
        ("Москва является столицей России, сказал он вчера.", None),
        ("პილოტის გვამი იდენტიფიცირებული იყო. ეს იყო ავიაციის მაიორი.", None),
    ],
)
def test_reject_reason(text, reason):
    assert reject_reason(text, LIMITS) == reason


def test_filter_texts_deduplicates_case_and_space_insensitively():
    kept, rejected = filter_texts(["The cat sat on the mat.", "the  cat sat on the Mat.", "Two"], LIMITS)
    assert kept == ["The cat sat on the mat."]
    assert rejected == Counter({Reject.DUPLICATE: 1, Reject.TOO_SHORT: 1})


# --- coverage selection --------------------------------------------------------------------------

def _plain_greedy(candidates, n):
    from gigagen.coverage import _gain, text_hash
    counts, chosen, left = Counter(), [], list(candidates)
    while left and len(chosen) < n:
        best = min(left, key=lambda c: (-_gain(c.symbols, counts), text_hash(c.text)))
        chosen.append(best)
        left.remove(best)
        counts.update(best.symbols)
    return chosen


def test_lazy_greedy_equals_plain_greedy():
    rng = np.random.default_rng(0)
    cands = [
        Candidate(f"s{i}", frozenset(int(x) for x in rng.choice(40, size=rng.integers(1, 12), p=None)))
        for i in range(120)
    ]
    assert greedy_order(cands, 60) == _plain_greedy(cands, 60)


def test_greedy_takes_rare_symbols_first():
    cands = [Candidate(f"common{i}", frozenset({1, 2})) for i in range(5)] + [Candidate("rare", frozenset({1, 99}))]
    order = greedy_order(cands, 3)
    assert order[1].text == "rare"


def test_held_out_is_shared_between_voices_reading_the_same_text():
    texts = [f"sentence number {i} is here" for i in range(100)]
    assert held_out(texts, 10) == held_out(list(reversed(texts)), 10)


def test_select_never_trains_on_any_voice_held_out():
    texts = [f"sentence number {i} is here" for i in range(60)]
    es = [Phonemized(t, frozenset({1, 2, i % 7})) for i, t in enumerate(texts)]
    es419 = [Phonemized(t, frozenset({1, 3})) for t in texts[10:]]
    lists, stats = select({"es": es, "es-419": es419}, train=1000, n_held_out=5)
    held = set(lists["es"].held_out) | set(lists["es-419"].held_out)
    for l in lists.values():
        assert not held & set(l.train)
    assert stats["es"]["train"] + len(held) >= 55


def test_select_drops_untokenizable():
    pool = [Phonemized("a b c d", None), Phonemized("e f g h", frozenset({1}))]
    lists, stats = select({"xx": pool}, train=10, n_held_out=0)
    assert lists["xx"].train == ("e f g h",)
    assert stats["xx"]["untokenizable"] == 1


# --- catalog -------------------------------------------------------------------------------------

def test_language_id_by_locale_then_espeak():
    assert language_id("es_MX", "es-419", REGIONS) == "es-419"
    assert language_id("es_AR", "es-419", REGIONS) == "es-AR"
    assert language_id("de_DE", "de", REGIONS) == "de"
    assert language_id(None, "es", REGIONS) == "es"


def test_parse_other_list_rebases_and_types():
    text = (
        "mms-az\tmms:az:/lap/bucket/az/model.mnn|/lap/bucket/az/tokens.txt\n"
        "mimic3-ko\tmimic3:ko:/lap/bucket/ko/ko_KO-kss_low.onnx.json\n"
        "kokoro-jm_kumo\tkokoro_ja:ja:/lap/bucket/dict/mucab.bin\n"
    )
    out = parse_other_list(text, Path("/lap/bucket"), Path("/box/b"))
    assert out["mms-az"] == (MmsSource(Path("/box/b/az/model.mnn"), Path("/box/b/az/tokens.txt")), "az")
    assert out["mimic3-ko"][0] == Mimic3Source(Path("/box/b/ko/ko_KO-kss_low.onnx.json"), Path("/box/b/ko/ko_KO-kss_low.mnn"))
    assert out["kokoro-jm_kumo"][0] == KokoroJaSource("jm_kumo", Path("/box/b/dict/mucab.bin"))
    with pytest.raises(CatalogError):
        parse_other_list("x\tmms:az:/elsewhere/model.mnn|/elsewhere/t.txt\n", Path("/lap/bucket"), Path("/box/b"))


TIERS = """lang,espeak,voice,speaker,human,audio,tier,condition,weight,articulation_rate
es,es,es_ES-sharvard-medium,F,pass,pass,good,clean,1.0,22.3
es,es-419,es_MX-ald-medium,0,pass,pass,good,clean,1.0,18.6
es,es,cv17-abc,0,pass,pass,good,clean,1.0,
"""


def _config(locale, speakers=None):
    return PiperVoiceConfig(PhonemeType.ESPEAK, 22050, locale.split("_")[0] if locale != "es_MX" else "es-419", locale,
                            len(speakers or {}) or 1, speakers or {}, {"_": [0]}, 0.667, 1.0, 0.8)


def test_build_speakers_multi_speaker_and_corpus():
    rows = parse_tiers(TIERS)
    piper = {
        "es_ES-sharvard-medium": PiperSource(Path("c"), Path("o"), Path("m"), _config("es_ES", {"M": 0, "F": 1})),
        "es_MX-ald-medium": PiperSource(Path("c"), Path("o"), Path("m"), _config("es_MX")),
    }
    speakers = {s.key: s for s in build_speakers(rows, piper, {}, REGIONS)}
    assert speakers["es_ES-sharvard-medium.F"].sid == 1
    assert speakers["es_MX-ald-medium"].language == "es-419"
    assert speakers["es_MX-ald-medium"].sid is None
    assert isinstance(speakers["cv17-abc"].source, CorpusSource)


def test_build_speakers_rejects_unknown_speaker():
    rows = parse_tiers(TIERS.replace(",F,", ",X,"))
    piper = {"es_ES-sharvard-medium": PiperSource(Path("c"), Path("o"), Path("m"), _config("es_ES", {"M": 0, "F": 1}))}
    with pytest.raises(CatalogError):
        build_speakers(rows[:1], piper, {}, REGIONS)


# --- plan ----------------------------------------------------------------------------------------

def _speaker(key, rate, tier=Tier.GOOD, lang="es", voice=None):
    return Speaker(key, voice or key, None, None, lang[:2], "es", lang, tier, Condition.CLEAN, 1.0, rate, CorpusSource())


def test_tempo_targets_the_good_median_of_the_language_id():
    speakers = [_speaker("slow", 18.0), _speaker("mid", 20.0), _speaker("fast", 22.0),
                _speaker("robot", 30.0, Tier.CLEAN_ROBOT), _speaker("robot2", 30.0, Tier.CLEAN_ROBOT),
                _speaker("gone", 5.0, Tier.EXCLUDED), _speaker("regional", 10.0, lang="es-AR"),
                _speaker("lonely_robot", 10.0, Tier.CLEAN_ROBOT, lang="ca")]
    raise_slow = tempo_factors(speakers, RatePolicy.RAISE_SLOW)
    assert raise_slow["slow"] == pytest.approx(0.9)
    assert raise_slow["fast"] == 1.0
    assert raise_slow["robot"] == 1.0
    assert raise_slow["regional"] == 1.0
    assert raise_slow["lonely_robot"] == 1.0
    assert "gone" not in raise_slow
    assert tempo_factors(speakers, RatePolicy.MATCH_MEDIAN)["fast"] == pytest.approx(1.1)


def test_hours_priority():
    hours = Hours({"good": 1.5}, 4.0, frozenset({"b"}), {"v": 0.2}, {"a": 0.1}, None)
    assert hours_for(_speaker("a", 1, voice="v"), hours) == 0.1
    assert hours_for(_speaker("b", 1, voice="v"), hours) == 4.0
    assert hours_for(_speaker("c", 1, voice="v"), hours) == 0.2
    assert hours_for(_speaker("d", 1), hours) == 1.5


def test_utterance_order_core_first_cycles_and_second_renders():
    sentences = tuple(f"s{i}" for i in range(50))
    seq = list(islice(utterance_order(sentences, 10, "spk", 1, 0.2), 200))
    firsts = [u for u in seq if u.render == 0]
    assert [u.text for u in firsts[:10]] == list(sentences[:10])
    assert len(firsts) == 50
    seconds = [u for u in seq[: len(firsts) + sum(1 for u in seq[:60] if u.render == 1)] if u.render == 1]
    assert all(u.text in sentences for u in seconds)
    ids = [(u.text, u.render) for u in seq]
    assert len(ids) == len(set(ids))
    assert any(u.render == 2 for u in seq)
    second_share = sum(1 for u in seq if u.render % 2 == 1) / len(seq)
    assert 0.05 < second_share < 0.35


def test_parts_partition_the_utterance_sequence():
    speaker = _speaker("spk", 20.0)
    hours = Hours({"good": 1.2}, 4.0, frozenset(), {}, {}, None)
    jobs = jobs_for(speaker, tuple(f"s{i}" for i in range(30)), hours, {"spk": 1.0}, 0.5, 5, 0.2, 1)
    assert len(jobs) == 3
    assert sum(j.budget_seconds for j in jobs) == pytest.approx(1.2 * 3600)
    taken = [list(islice(part_utterances(j), 20)) for j in jobs]
    whole = list(islice(utterance_order(jobs[0].sentences, 5, "spk", 1, 0.2), 60))
    assert sorted((u.text, u.render) for t in taken for u in t) == sorted((u.text, u.render) for u in whole)


def test_cap_seconds_limits_budget():
    hours = Hours({"good": 1.5}, 4.0, frozenset(), {}, {}, 20.0)
    [job] = jobs_for(_speaker("spk", 20.0), ("a b c d",), hours, {"spk": 1.0}, 0.5, 1, 0.2, 1)
    assert job.budget_seconds == 20.0


# --- alignment -----------------------------------------------------------------------------------

def test_rescale_durations_keeps_boundaries_within_half_a_frame():
    rng = np.random.default_rng(1)
    d = [int(x) for x in rng.integers(0, 15, size=200)]
    for rate in (16000, 24000, 44100):
        out = rescale_durations(d, rate)
        ratio = 22050 / rate
        assert sum(out) == round(sum(d) * ratio + 1e-9) or abs(sum(out) - sum(d) * ratio) <= 0.5
        assert all(x >= 0 for x in out)
        for k in range(len(d)):
            assert abs(sum(out[: k + 1]) - sum(d[: k + 1]) * ratio) <= 0.5 + 1e-9
    assert rescale_durations(d, 22050) == d


def test_fit_frames():
    a = np.ones(1000, dtype=np.float32)
    assert fit_frames(a, 4).size == 1024
    assert fit_frames(np.ones(1100, dtype=np.float32), 4).size == 1024
    with pytest.raises(AlignmentError):
        fit_frames(a, 6)


# --- quality -------------------------------------------------------------------------------------

def test_internal_silence_ignores_edges():
    rate = 22050
    tone = 0.5 * np.sin(np.linspace(0, 2000, rate)).astype(np.float32)
    quiet = np.zeros(rate, dtype=np.float32)
    assert longest_internal_silence(np.concatenate([quiet, tone, quiet]), rate, 0.02, -45) == 0.0
    gap = longest_internal_silence(np.concatenate([tone, quiet, quiet, tone]), rate, 0.02, -45)
    assert gap == pytest.approx(2.0, abs=0.05)


def test_duration_outliers_flags_a_held_phoneme():
    rng = np.random.default_rng(2)
    timings = []
    for _ in range(40):
        body = [int(x) for x in rng.integers(4, 8, size=10)]
        ids, durs = [1, 0], [2, 1]
        for b in body:
            ids += [7, 0]
            durs += [b, 1]
        timings.append(TeacherTiming(tuple(ids + [2]), tuple(durs + [3])))
    held = list(timings[5].durations)
    held[4] = 90
    timings[5] = TeacherTiming(timings[5].ids, tuple(held))
    assert set(duration_outliers(timings, frozenset({7}), 4.0, 30)) == {5}
    assert duration_outliers(timings, frozenset({8}), 4.0, 30) == {}
    assert duration_outliers(timings[:2], frozenset({7}), 4.0, 30) == {}


# --- packaging -----------------------------------------------------------------------------------

def test_assign_shards_respects_size():
    items = [Item({"id": f"i{n}"}, Path(f"/x/{n}"), 300) for n in range(10)]
    shards = assign_shards(items, 1000, 1)
    assert [len(s) for s in shards] == [3, 3, 3, 1]
    assert sorted(i.record["id"] for s in shards for i in s) == sorted(i.record["id"] for i in items)


def test_teacher_reports_flag_heavy_losers():
    stats = [
        JobStats("a.p000", "a", "a", "es", 100, 500.0, 85, 400.0, {"internal_silence": 15}, 0),
        JobStats("b.p000", "b", "b", "es", 100, 500.0, 95, 480.0, {"duration_outlier": 5}, 0),
    ]
    reports = {r.teacher: r for r in teacher_reports(stats, 0.10)}
    assert reports["a"].flagged and not reports["b"].flagged
