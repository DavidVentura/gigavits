"""Durations across a sample-rate change (pure)."""
from __future__ import annotations

HOP = 256
TARGET_RATE = 22050


class AlignmentError(ValueError):
    pass


def rescale_durations(durations: list[int], rate: int) -> list[int]:
    """Frames of HOP samples at `rate` -> frames of HOP samples at TARGET_RATE.

    Phoneme boundaries are rounded (half up) in the new frame grid, so each boundary moves by at
    most half a frame and the total is the rounded total length.
    """
    out, previous, cumulative = [], 0, 0
    for d in durations:
        cumulative += d
        boundary = (2 * cumulative * TARGET_RATE + rate) // (2 * rate)
        out.append(boundary - previous)
        previous = boundary
    return out
