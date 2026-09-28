"""Render quality filter (PLAN 2.4): duration outliers and long internal silences (pure)."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Drop(Enum):
    DURATION_OUTLIER = "duration_outlier"
    INTERNAL_SILENCE = "internal_silence"


@dataclass(frozen=True)
class TeacherTiming:
    """Teacher input IDs and the frames the teacher gave each, as rendered."""

    ids: tuple[int, ...]
    durations: tuple[int, ...]


def longest_internal_silence(audio: np.ndarray, rate: int, frame_s: float, silence_db: float) -> float:
    """Longest run of silent frames between the first and the last non-silent frame, in seconds."""
    frame = max(1, round(frame_s * rate))
    n = audio.size // frame
    if n == 0:
        return 0.0
    frames = audio[: n * frame].astype(np.float64).reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    silent = 20 * np.log10(np.maximum(rms, 1e-10)) < silence_db
    voiced = np.flatnonzero(~silent)
    if voiced.size == 0:
        return n * frame / rate
    inner = silent[voiced[0] : voiced[-1] + 1]
    longest, run = 0, 0
    for s in inner:
        run = run + 1 if s else 0
        longest = max(longest, run)
    return longest * frame / rate


def symbol_positions(length: int) -> range:
    """Positions of real symbols in [bos, pad, (symbol, pad)*, eos]; pads, BOS and EOS carry
    context-dependent silence and are left to the silence check."""
    return range(2, length - 1, 2)


def duration_outliers(
    timings: list[TeacherTiming], judged: frozenset[int], z_max: float, min_count: int
) -> dict[int, float]:
    """Renders holding some phoneme more than z_max standard deviations above that phoneme's mean
    over all given renders of one teacher, as index -> largest z.

    Only `judged` IDs count: spaces and punctuation carry phrase pauses, which the silence check
    covers. z is taken on log(1 + frames): durations are right-skewed, and with 16 kHz voices' coarse
    frames a normally lengthened phrase-final vowel would otherwise score like a glitch. Phonemes
    seen fewer than min_count times have no usable spread and are not judged.
    """
    values: dict[int, list[float]] = defaultdict(list)
    for t in timings:
        for i in symbol_positions(len(t.ids)):
            if t.ids[i] in judged:
                values[t.ids[i]].append(math.log1p(t.durations[i]))
    stats = {}
    for symbol, v in values.items():
        if len(v) < min_count:
            continue
        arr = np.asarray(v, dtype=np.float64)
        std = arr.std()
        if std > 0:
            stats[symbol] = (arr.mean(), std)
    out = {}
    for index, t in enumerate(timings):
        worst = max(
            (
                (math.log1p(t.durations[i]) - stats[t.ids[i]][0]) / stats[t.ids[i]][1]
                for i in symbol_positions(len(t.ids))
                if t.ids[i] in stats
            ),
            default=0.0,
        )
        if worst > z_max:
            out[index] = worst
    return out
