import pytest

from criteria import (
    AbAnswer,
    Inputs,
    Rating,
    Status,
    Verdict,
    evaluate,
    needs_ears,
    parse_spec,
    parse_verdicts,
    verdict_voice_name,
)
from languages import load_languages
from manifest import VoiceKey
from scores import ScoreRow

DE = VoiceKey("de_DE-thorsten-medium", "0")
ES = VoiceKey("es_AR-daniela-high", "0")
SERENA = VoiceKey("it_IT-serena-medium", "0")
PAOLA = VoiceKey("it_IT-paola-medium", "0")
DAVEFX = VoiceKey("es_ES-davefx-medium", "0")

SPEC = parse_spec({
    "lineup": [
        {"voice": DE.voice, "speaker": "0", "language": "de"},
        {"voice": ES.voice, "speaker": "0", "language": "es-419"},
    ],
    "native_rating_languages": ["es"],
    "weak_language": "it",
    "weak_reference": str(SERENA),
})


def rows(key, espeak, n=3, reference=None, **values):
    return [
        ScoreRow(voice=key.voice, speaker=key.speaker, espeak=espeak, idx=i, text=f"{espeak}-{i}", path=f"/{key.voice}{espeak}{i}.wav",
                 duration_s=3.0, speaker_reference=None if reference is None else f"ref/{reference}", **values)
        for i in range(n)
    ]


def originals():
    return (
        rows(DE, "de", cer_edits=2, cer_chars=100, UTMOS=3.2, lid_target=0.99)
        + rows(ES, "es-419", cer_edits=3, cer_chars=100, UTMOS=2.8, lid_target=0.9)
        + rows(DAVEFX, "es", cer_edits=4, cer_chars=100, UTMOS=3.0, lid_target=0.95)
        + rows(SERENA, "it", per_edits=10, per_units=100, UTMOS=3.4, lid_target=0.98)
        + rows(PAOLA, "it", per_edits=12, per_units=100, UTMOS=3.1, lid_target=0.97)
    )


def candidate(de_cer=3, de_utmos=3.15, es_lid=0.9, it_per=14):
    return (
        rows(DE, "de", reference=DE, cer_edits=de_cer, cer_chars=100, UTMOS=de_utmos, speaker_similarity=0.85, lid_target=0.99)
        + rows(ES, "es-419", reference=ES, cer_edits=3, cer_chars=100, UTMOS=2.9, speaker_similarity=0.79)
        + rows(DE, "es", reference=DE, cer_edits=5, cer_chars=100, UTMOS=3.1, speaker_similarity=0.72, lid_target=es_lid)
        + rows(DE, "it", reference=DE, per_edits=it_per, per_units=100, UTMOS=3.0, speaker_similarity=0.70, lid_target=0.9)
        + rows(ES, "it", reference=ES, per_edits=20, per_units=100, UTMOS=2.6, speaker_similarity=0.6, lid_target=0.9)
    )


def inputs(**kw):
    base = dict(
        spec=SPEC, languages=load_languages(), originals=originals(), candidate=candidate(),
        original_human={DE: "pass", ES: "pass"},
        original_human_by_language={"de": ["pass"], "es": ["pass", "fail"], "es-419": ["pass"], "it": ["fail"]},
        verdicts=None, ab=None, ratings=None, runtime=None,
    )
    return Inputs(**(base | kw))


def find(results, test, voice="*", language="*"):
    matches = [r for r in results if r.test == test and r.voice == voice and r.language == language]
    assert len(matches) == 1, (test, voice, language, matches)
    return matches[0]


def test_own_language_metrics():
    results = evaluate(inputs())
    assert find(results, "own: Whisper CER vs original", str(DE), "de").status is Status.PASS
    assert find(results, "own: speaker similarity to original", str(DE), "de").status is Status.PASS
    assert find(results, "own: speaker similarity to original", str(ES), "es-419").status is Status.FAIL
    assert find(results, "own: UTMOS vs original (recording quality only)", str(DE), "de").status is Status.PASS
    assert find(results, "own: blind A/B vs original", str(DE), "de").status is Status.NOT_MEASURED
    assert find(results, "own: human enough (verdict page)", str(DE), "de").status is Status.NOT_MEASURED


def test_cer_margin_is_two_points():
    assert find(evaluate(inputs(candidate=candidate(de_cer=4))), "own: Whisper CER vs original", str(DE), "de").status is Status.PASS
    assert find(evaluate(inputs(candidate=candidate(de_cer=5))), "own: Whisper CER vs original", str(DE), "de").status is Status.FAIL


def test_utmos_margin_is_one_tenth_below():
    assert find(evaluate(inputs(candidate=candidate(de_utmos=3.05))), "own: UTMOS vs original (recording quality only)", str(DE), "de").status is Status.FAIL


def test_cer_applicability_follows_the_allow_list():
    results = evaluate(inputs())
    assert find(results, "cross: Whisper CER vs best original", str(DE), "it").status is Status.NOT_MEASURED
    unscored = [r for r in candidate() if r.espeak != "de"] + [
        r.__class__(**(r.__dict__ | {"cer_edits": None, "cer_chars": None})) for r in candidate() if r.espeak == "de"]
    assert find(evaluate(inputs(candidate=unscored)), "own: Whisper CER vs original", str(DE), "de").status is Status.NOT_MEASURED
    assert find(results, "cross: language-ID confidence", str(DE), "it").status is Status.PASS


def test_cross_language():
    results = evaluate(inputs())
    cer = find(results, "cross: Whisper CER vs best original", str(DE), "es")
    assert cer.reference == pytest.approx(0.04) and cer.status is Status.PASS
    lid = find(results, "cross: language-ID confidence", str(DE), "es")
    assert lid.reference == pytest.approx(0.95) and lid.status is Status.PASS
    assert find(evaluate(inputs(candidate=candidate(es_lid=0.8))), "cross: language-ID confidence", str(DE), "es").status is Status.FAIL
    assert find(results, "cross: best lineup UTMOS vs best original (within language)", language="es").status is Status.PASS
    assert find(results, "cross: best lineup UTMOS vs best original (within language)", language="it").status is Status.FAIL
    assert find(results, "cross: speaker similarity to own-language output", str(DE)).status is Status.PASS
    assert find(results, "cross: speaker similarity to own-language output", str(ES)).status is Status.FAIL
    assert find(results, "cross: some lineup voice human enough", language="it").status is Status.PASS
    assert find(results, "cross: some lineup voice human enough", language="es").status is Status.NOT_MEASURED
    assert find(results, "cross: native rating", language="es").status is Status.NOT_MEASURED


def test_weak_language():
    results = evaluate(inputs())
    assert find(results, "weak: phoneme error vs reference", str(DE), "it").status is Status.PASS
    assert find(results, "weak: phoneme error vs reference", str(ES), "it").status is Status.FAIL
    assert find(results, "weak: UTMOS vs own-language UTMOS", str(DE), "it").status is Status.PASS
    assert find(results, "weak: UTMOS vs own-language UTMOS", str(ES), "it").status is Status.FAIL
    assert find(evaluate(inputs(candidate=candidate(it_per=16))), "weak: phoneme error vs reference", str(DE), "it").status is Status.FAIL


def test_ab_counts_ties_half_and_needs_thirty():
    def answers(n_orig, n_tie, n_new):
        out, i = [], 0
        for answer, count in (("A", n_orig), ("tie", n_tie), ("B", n_new)):
            for _ in range(count):
                out.append(AbAnswer("me", DE, str(i), "A", answer))
                i += 1
        return out

    result = find(evaluate(inputs(ab=answers(15, 10, 5))), "own: blind A/B vs original", str(DE), "de")
    assert result.measured == pytest.approx(20 / 30) and result.status is Status.FAIL
    assert find(evaluate(inputs(ab=answers(12, 12, 6))), "own: blind A/B vs original", str(DE), "de").status is Status.PASS
    assert find(evaluate(inputs(ab=answers(5, 5, 5))), "own: blind A/B vs original", str(DE), "de").status is Status.NOT_MEASURED


def test_verdicts_and_ratings():
    verdicts = [Verdict(DE, "de", "fail"), Verdict(ES, "es-419", "pass"), Verdict(DE, "es", "pass")]
    ratings = [Rating("friend", "es", DE, str(i), s, "") for i, s in enumerate([4, 3, 4])]
    results = evaluate(inputs(verdicts=verdicts, ratings=ratings))
    assert find(results, "own: human enough (verdict page)", str(DE), "de").status is Status.FAIL
    assert find(results, "own: human enough (verdict page)", str(ES), "es-419").status is Status.PASS
    assert find(results, "cross: some lineup voice human enough", language="es").status is Status.PASS
    assert find(results, "cross: native rating", language="es").measured == pytest.approx(11 / 3)


def test_original_that_failed_human_passes_trivially():
    results = evaluate(inputs(original_human={DE: "fail", ES: "pass"}))
    assert find(results, "own: human enough (verdict page)", str(DE), "de").status is Status.PASS


def test_runtime():
    results = evaluate(inputs(runtime={"base": 1.08, "wide": 2.1}))
    assert find(results, "runtime: CPU per audio second vs medium voice", "base").status is Status.PASS
    assert find(results, "runtime: CPU per audio second vs medium voice", "wide").status is Status.FAIL


def test_candidate_outside_lineup_fails():
    with pytest.raises(ValueError, match="outside the lineup"):
        evaluate(inputs(candidate=candidate() + rows(PAOLA, "it")))


def test_wrong_speaker_reference_fails():
    bad = rows(DE, "de", reference=PAOLA, UTMOS=3.0) + candidate()[3:]
    with pytest.raises(ValueError, match="speaker similarity against"):
        evaluate(inputs(candidate=bad))


def test_needs_ears_lists_missing_listening_and_low_utmos():
    data = inputs(ratings=[Rating("friend", "es", DE, "1", 3, "the r is English")])
    ears = needs_ears(data, evaluate(data))
    tasks = {(e.task, e.voice, e.language) for e in ears}
    assert ("human enough?", str(DE), "de") in tasks
    assert ("blind A/B", str(ES), "es-419") in tasks
    assert ("read note: sound clearly wrong?", str(DE), "es") in tasks
    assert ("re-listen: low recording quality", str(ES), "it") in tasks
    assert ("carried-over narrow band or crackle?", str(DE), "it") in tasks


def test_verdict_names_roundtrip(tmp_path):
    name = verdict_voice_name("run1-50k", DE, "es-419")
    path = tmp_path / "verdicts.csv"
    path.write_text(f'lang,voice,speaker,human,audio\n"es","{name}","0","pass",""\n"es","{DAVEFX.voice}","0","fail",""\n')
    assert parse_verdicts(path, "run1-50k") == [Verdict(DE, "es-419", "pass")]
    assert parse_verdicts(path, "other") == []


def test_spec_must_be_complete():
    with pytest.raises(ValueError):
        parse_spec({"lineup": []})
