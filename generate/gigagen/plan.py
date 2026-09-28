"""Which speakers render what: hours per speaker, tempo, utterance order and job split (pure)."""
from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count, islice

from .catalog import CorpusSource, Speaker, Tier
from .config import Hours, RatePolicy, Selection
from .coverage import text_hash


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Utterance:
    text: str
    render: int  # 0, 1, ... per (speaker, text); odd renders are the second-seed copies


@dataclass(frozen=True)
class Job:
    job_id: str
    speaker: Speaker
    part: int
    parts: int
    budget_seconds: float
    # multiplies the teacher's length scale (Kokoro/MMS/...: divides speed)
    tempo: float
    seed: int
    sentences: tuple[str, ...]
    core: int
    second_render_fraction: float


def unit_hash(*parts: object) -> float:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def int_hash(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()[:4], "big")


def utterance_id(speaker_key: str, u: Utterance) -> str:
    return f"{speaker_key}-{text_hash(u.text)[:12]}-r{u.render}"


def selected(speakers: list[Speaker], selection: Selection) -> tuple[list[Speaker], list[Speaker]]:
    """(to render, recorded-speech speakers left to the corpus loaders)."""
    chosen = [
        s for s in speakers
        if s.tier is not Tier.EXCLUDED
        and s.voice not in selection.exclude_voices
        and s.key not in selection.exclude_speakers
        and s.espeak not in selection.exclude_espeak
        and (not selection.voices or s.voice in selection.voices)
    ]
    unknown = (selection.voices | selection.exclude_voices) - {s.voice for s in speakers}
    unknown |= selection.exclude_speakers - {s.key for s in speakers}
    if unknown:
        raise PlanError(f"selection names voices or speakers not in tiers.csv: {sorted(unknown)}")
    corpus = [s for s in chosen if isinstance(s.source, CorpusSource)]
    return [s for s in chosen if not isinstance(s.source, CorpusSource)], corpus


def hours_for(speaker: Speaker, hours: Hours) -> float:
    if speaker.key in hours.lineup_speakers:
        return hours.lineup
    if speaker.key in hours.by_speaker:
        return hours.by_speaker[speaker.key]
    if speaker.voice in hours.by_voice:
        return hours.by_voice[speaker.voice]
    return hours.by_tier[speaker.tier.value]


def tempo_factors(speakers: list[Speaker], policy: RatePolicy) -> dict[str, float]:
    """Length-scale multiplier per speaker key from articulation rates (phonemes per second).

    The target is the median rate of the language ID's good-tier speakers, over tiers.csv rather
    than what a run selects, so a laptop run and a box run render a voice at the same tempo. A
    language with no good speaker of measured rate has no target, and its voices keep their tempo.
    """
    rates: dict[str, list[float]] = {}
    for s in speakers:
        if s.tier is Tier.GOOD and s.articulation_rate is not None:
            rates.setdefault(s.language, []).append(s.articulation_rate)
    medians = {language: statistics.median(r) for language, r in rates.items()}
    out = {}
    for s in speakers:
        if s.tier is Tier.EXCLUDED or s.articulation_rate is None:
            continue
        if s.language not in medians:
            out[s.key] = 1.0
            continue
        ratio = s.articulation_rate / medians[s.language]
        match policy:
            case RatePolicy.RAISE_SLOW:
                out[s.key] = min(1.0, ratio)
            case RatePolicy.MATCH_MEDIAN:
                out[s.key] = ratio
    return out


def utterance_order(sentences: tuple[str, ...], core: int, speaker_key: str, seed: int, second_fraction: float) -> Iterator[Utterance]:
    """Endless: the coverage head first for every speaker, then the rest in a per-speaker order so
    speakers of one language spread over the pool; cycles with new renders once exhausted."""
    if not sentences:
        raise PlanError(f"{speaker_key}: no training sentences")
    rest = sorted(sentences[core:], key=lambda t: unit_hash(seed, speaker_key, t))
    order = sentences[:core] + tuple(rest)
    for cycle in count():
        for text in order:
            yield Utterance(text, 2 * cycle)
            if unit_hash(seed, speaker_key, text, cycle, "second") < second_fraction:
                yield Utterance(text, 2 * cycle + 1)


def part_utterances(job: Job) -> Iterator[Utterance]:
    order = utterance_order(job.sentences, job.core, job.speaker.key, job.seed, job.second_render_fraction)
    return islice(order, job.part, None, job.parts)


def jobs_for(
    speaker: Speaker,
    sentences: tuple[str, ...],
    hours: Hours,
    tempo: dict[str, float],
    part_hours: float,
    core: int,
    second_fraction: float,
    seed: int,
) -> list[Job]:
    if speaker.key not in tempo:
        raise PlanError(f"{speaker.key}: no articulation rate in tiers.csv, cannot set its tempo")
    seconds = hours_for(speaker, hours) * 3600
    if hours.cap_seconds is not None:
        seconds = min(seconds, hours.cap_seconds)
    parts = max(1, math.ceil(seconds / (part_hours * 3600)))
    return [
        Job(
            job_id=f"{speaker.key}.p{p:03d}",
            speaker=speaker,
            part=p,
            parts=parts,
            budget_seconds=seconds / parts,
            tempo=tempo[speaker.key],
            seed=seed,
            sentences=sentences,
            core=core,
            second_render_fraction=second_fraction,
        )
        for p in range(parts)
    ]
