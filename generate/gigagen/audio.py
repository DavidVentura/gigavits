"""Audio conversions shared by every teacher path."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile
import soxr

from .align import HOP, TARGET_RATE, AlignmentError


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    """piper-rs `normalize_audio`: divide by the peak (at least 0.01), as the app plays every voice."""
    peak = max(0.01, float(np.max(np.abs(audio))))
    return (audio / peak).astype(np.float32)


def to_target_rate(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == TARGET_RATE:
        return audio
    return soxr.resample(audio, rate, TARGET_RATE, quality="VHQ").astype(np.float32)


def fit_frames(audio: np.ndarray, frames: int) -> np.ndarray:
    """Pads or cuts the end so the audio is exactly frames * HOP samples. Resampling the teacher's
    exact frame count leaves at most one frame of rounding at the end (trailing EOS silence)."""
    target = frames * HOP
    diff = target - audio.size
    if abs(diff) > HOP:
        raise AlignmentError(f"audio has {audio.size} samples, durations say {target}")
    if diff >= 0:
        return np.concatenate([audio, np.zeros(diff, dtype=audio.dtype)])
    return audio[:target]


def write_flac(path: Path, audio: np.ndarray) -> int:
    soundfile.write(path, np.clip(audio, -1.0, 1.0), TARGET_RATE, format="FLAC", subtype="PCM_16")
    return path.stat().st_size
