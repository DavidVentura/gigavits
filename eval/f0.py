"""F0 contour correlation against a reference rendering of the same sentence.

Frame-by-frame comparison is only meaningful when both renders share durations (the new model
rendered with the original's per-phoneme durations), so lengths must agree within a tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import librosa
import numpy as np

from config import F0Config

HOP = 160
FRAME = 1024


class F0Status(Enum):
    OK = "ok"
    LENGTH_MISMATCH = "length_mismatch"
    TOO_FEW_VOICED = "too_few_voiced"


@dataclass(frozen=True)
class F0Result:
    status: F0Status
    correlation: float | None


def f0_track(audio: np.ndarray, sample_rate: int, config: F0Config) -> np.ndarray:
    """F0 in Hz per frame, NaN where unvoiced."""
    f0, voiced, _ = librosa.pyin(
        audio, fmin=config.fmin_hz, fmax=config.fmax_hz, sr=sample_rate, frame_length=FRAME, hop_length=HOP
    )
    return np.where(voiced, f0, np.nan)


def contour_correlation(track: np.ndarray, reference: np.ndarray, min_voiced_frames: int) -> F0Result:
    n = min(len(track), len(reference))
    a, b = track[:n], reference[:n]
    both = ~np.isnan(a) & ~np.isnan(b)
    if both.sum() < min_voiced_frames:
        return F0Result(F0Status.TOO_FEW_VOICED, None)
    return F0Result(F0Status.OK, float(np.corrcoef(a[both], b[both])[0, 1]))


def f0_correlation(audio: np.ndarray, reference: np.ndarray, sample_rate: int, config: F0Config) -> F0Result:
    mismatch_s = abs(len(audio) - len(reference)) / sample_rate
    if mismatch_s > config.max_length_mismatch_s:
        return F0Result(F0Status.LENGTH_MISMATCH, None)
    return contour_correlation(
        f0_track(audio, sample_rate, config), f0_track(reference, sample_rate, config), config.min_voiced_frames
    )
