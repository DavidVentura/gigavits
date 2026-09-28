from pathlib import Path

import pytest

from config import F0Config, Hub, WhisperConfig, parse_config
from manifest import parse_manifest
from repo import ROOT
from score import Context, make_context, score_utterance

SAMPLES = ROOT / "voices" / "samples" / "manifest.tsv"


def utterance(voice: str, idx: int = 0):
    return next(u for u in parse_manifest(SAMPLES) if u.key.voice == voice and u.idx == idx)


def context_with_fake_whisper(transcripts: list[str]) -> tuple[Context, list[str]]:
    config = parse_config({"threads": 1}, Path("."))
    config = config.__class__(**(config.__dict__ | {"cer": WhisperConfig(Hub("small"), "cpu", "int8", 5)}))
    ctx = make_context(config, {})
    calls = []

    def fake(audio, language):
        calls.append(language)
        return transcripts.pop(0)

    ctx.models.__dict__["transcriber"] = fake
    return ctx, calls


def test_cer_uses_the_whisper_code_and_the_allow_list():
    de = utterance("de_DE-thorsten-medium")
    ctx, calls = context_with_fake_whisper([de.text.upper().replace(",", "")])
    row = score_utterance(de, None, ctx)
    assert calls == ["de"]
    assert row.cer_edits == 0 and row.cer_chars > 50 and row.duration_s > 1
    fi = utterance("fi_FI-harri-medium")
    row = score_utterance(fi, None, ctx)
    assert row.cer_edits is None and calls == ["de"]


def test_f0_only_with_a_reference():
    de = utterance("de_DE-thorsten-medium")
    config = parse_config({"threads": 1}, Path("."))
    config = config.__class__(**(config.__dict__ | {"f0": F0Config(50, 600, 0.02, 20)}))
    ctx = make_context(config, {})
    assert score_utterance(de, None, ctx).f0_status is None
    same = de.__class__(**(de.__dict__ | {"reference_wav": de.path}))
    row = score_utterance(same, None, ctx)
    assert row.f0_status == "ok" and row.f0_corr == pytest.approx(1.0)
