import numpy as np
import pytest

from config import F0Config
from f0 import F0Status, contour_correlation, f0_correlation
from models import ctc_collapse


def test_ctc_collapse_merges_repeats_and_drops_blanks():
    assert ctc_collapse([0, 5, 5, 0, 5, 7, 7, 0], blank=0) == [5, 5, 7]
    assert ctc_collapse([], blank=0) == []


def test_contour_correlation_uses_frames_voiced_in_both():
    a = np.array([100, 110, 120, np.nan, 140, 150.0])
    b = np.array([200, 220, 240, 260, np.nan, 300.0])
    result = contour_correlation(a, b, min_voiced_frames=3)
    assert result.status is F0Status.OK and result.correlation == pytest.approx(1.0)
    assert contour_correlation(a, b, min_voiced_frames=5).status is F0Status.TOO_FEW_VOICED


def test_length_mismatch_is_reported_not_measured():
    config = F0Config(50, 600, 0.02, 5)
    result = f0_correlation(np.zeros(16000, np.float32), np.zeros(17000, np.float32), 16000, config)
    assert result == (result.__class__(F0Status.LENGTH_MISMATCH, None))


def test_identical_tones_correlate():
    sr = 16000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * (150 + 50 * t) * t)).astype(np.float32)
    result = f0_correlation(tone, tone, sr, F0Config(50, 600, 0.02, 20))
    assert result.status is F0Status.OK and result.correlation > 0.99
