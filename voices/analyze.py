"""Per-speaker acoustic metrics for the voice tiering pass, and the rating page.

Metrics are heuristics to be calibrated against the owner's ratings, not verdicts.
"""
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window

FRAME_S = 0.025
HOP_S = 0.010
F0_MIN, F0_MAX = 70.0, 400.0


@dataclass(frozen=True)
class Clip:
    voice: str
    speaker: str
    espeak: str
    file: str
    phonemes: int


@dataclass(frozen=True)
class SpeakerMetrics:
    voice: str
    speaker: str
    espeak: str
    sample_rate: int
    bandwidth_khz: float
    hf_energy_db: float
    snr_db: float
    harmonicity: float
    f0_median_hz: float
    f0_range_st: float
    f0_step_st: float
    phonemes_per_s: float
    articulation_rate: float
    mel_gv_db: float
    spectral_flux: float
    flux_variability: float
    shimmer_db: float


def read_manifest(path: Path) -> list[Clip]:
    with path.open() as f:
        return [
            Clip(r["voice"], r["speaker"], r["espeak"], r["file"], int(r["phonemes"]))
            for r in csv.DictReader(f, delimiter="\t")
        ]


def load(path: Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)
    return sr, x.astype(np.float64) / 32768.0


def frames(x: np.ndarray, sr: int) -> np.ndarray:
    n, hop = int(FRAME_S * sr), int(HOP_S * sr)
    count = 1 + max(0, (len(x) - n) // hop)
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


def frame_db(fr: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.mean(fr**2, axis=1) + 1e-12)


def spectral_stats(fr: np.ndarray, speech: np.ndarray, sr: int) -> tuple[float, float]:
    spec = np.abs(np.fft.rfft(fr[speech] * get_window("hann", fr.shape[1]), axis=1)) ** 2
    power = spec.sum(axis=0)
    freqs = np.fft.rfftfreq(fr.shape[1], 1 / sr)
    # The spectral edge, where the long-term spectrum drops 50 dB below its peak, separates
    # audio that was recorded or upsampled from 16 kHz from true full-band audio.
    smooth = np.convolve(10 * np.log10(power + 1e-20), np.ones(5) / 5, mode="same")
    edge = freqs[np.flatnonzero(smooth > smooth.max() - 50)[-1]]
    hf = power[freqs >= 5000].sum() / power.sum()
    return edge / 1000, 10 * np.log10(hf + 1e-12)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    def to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    edges = 700 * (10 ** (np.linspace(to_mel(80), to_mel(min(8000, sr / 2)), n_mels + 2) / 2595) - 1)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    bank = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        bank[m] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return bank


def richness(fr: np.ndarray, voiced: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    """Over-smoothing measures: synthetic speech varies less, frame to frame and band to band.

    Bands stop at 8 kHz so 16 kHz voices are measured on the same range as full-band ones.
    """
    spec = np.abs(np.fft.rfft(fr[voiced] * get_window("hann", fr.shape[1]), axis=1)) ** 2
    mel_db = 10 * np.log10(spec @ mel_filterbank(sr, fr.shape[1]).T + 1e-10)
    gv = float(np.mean(np.std(mel_db, axis=0)))
    flux = np.sqrt(np.mean(np.diff(mel_db, axis=0) ** 2, axis=1))
    frame_db_voiced = 10 * np.log10(np.mean(fr[voiced] ** 2, axis=1) + 1e-12)
    shimmer = float(np.median(np.abs(np.diff(frame_db_voiced))))
    return gv, float(np.median(flux)), float(np.std(flux) / (np.mean(flux) + 1e-9)), shimmer


def f0_track(fr: np.ndarray, voiced_mask: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    f0 = np.full(len(fr), np.nan)
    peak = np.zeros(len(fr))
    for i in np.flatnonzero(voiced_mask):
        x = fr[i] - fr[i].mean()
        ac = np.correlate(x, x, "full")[len(x) - 1 :]
        if ac[0] <= 0 or hi >= len(ac):
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        peak[i] = ac[lag] / ac[0]
        if peak[i] > 0.5:
            f0[i] = sr / lag
    return f0, peak


def speaker_metrics(clips: list[Clip], sample_dir: Path) -> SpeakerMetrics:
    per_clip = []
    for clip in clips:
        sr, x = load(sample_dir / clip.file)
        fr = frames(x, sr)
        db = frame_db(fr)
        loud, quiet = np.percentile(db, 95), np.percentile(db, 10)
        speech = db > quiet + 0.4 * (loud - quiet)
        active_s = (np.flatnonzero(speech)[-1] - np.flatnonzero(speech)[0] + 1) * HOP_S
        bandwidth, hf = spectral_stats(fr, speech, sr)
        f0, peak = f0_track(fr, speech, sr)
        speaking_s = speech.sum() * HOP_S
        voiced = speech & ~np.isnan(f0_track(fr, speech, sr)[0])
        rich = richness(fr, voiced, sr)
        per_clip.append((sr, bandwidth, hf, loud - quiet, quiet - loud, peak[speech], f0, clip.phonemes / active_s, clip.phonemes / speaking_s, rich))
    first = clips[0]
    f0_all = np.concatenate([c[6] for c in per_clip])
    semitones = 12 * np.log2(f0_all[~np.isnan(f0_all)] / 100)
    steps = np.concatenate([np.abs(np.diff(12 * np.log2(c[6] / 100))) for c in per_clip])
    steps = steps[~np.isnan(steps)]
    return SpeakerMetrics(
        voice=first.voice,
        speaker=first.speaker,
        espeak=first.espeak,
        sample_rate=per_clip[0][0],
        bandwidth_khz=round(float(np.median([c[1] for c in per_clip])), 2),
        hf_energy_db=round(float(np.median([c[2] for c in per_clip])), 1),
        snr_db=round(float(np.median([c[3] for c in per_clip])), 1),
        harmonicity=round(float(np.mean(np.concatenate([c[5] for c in per_clip]))), 3),
        f0_median_hz=round(float(100 * 2 ** (np.median(semitones) / 12)), 0),
        f0_range_st=round(float(np.percentile(semitones, 90) - np.percentile(semitones, 10)), 1),
        f0_step_st=round(float(np.median(steps)), 3),
        phonemes_per_s=round(float(np.median([c[7] for c in per_clip])), 1),
        articulation_rate=round(float(np.median([c[8] for c in per_clip])), 1),
        mel_gv_db=round(float(np.median([c[9][0] for c in per_clip])), 2),
        spectral_flux=round(float(np.median([c[9][1] for c in per_clip])), 2),
        flux_variability=round(float(np.median([c[9][2] for c in per_clip])), 3),
        shimmer_db=round(float(np.median([c[9][3] for c in per_clip])), 2),
    )


@dataclass(frozen=True)
class Thresholds:
    snr_db: float
    f0_range_st: float


def thresholds(metrics: list[SpeakerMetrics]) -> Thresholds:
    # Relative to the catalogue until the owner's ratings calibrate absolute values.
    return Thresholds(
        snr_db=float(np.percentile([m.snr_db for m in metrics], 15)),
        f0_range_st=float(np.percentile([m.f0_range_st for m in metrics], 15)),
    )


def flags(m: SpeakerMetrics, language_rate: float, t: Thresholds) -> list[str]:
    out = []
    if m.sample_rate < 22050:
        out.append("16 kHz")
    elif m.bandwidth_khz < 8.5:
        out.append("band-limited")
    if m.snr_db < t.snr_db:
        out.append("noisy?")
    if m.f0_range_st < t.f0_range_st:
        out.append("monotone?")
    if m.phonemes_per_s < 0.8 * language_rate:
        out.append("slow?")
    if m.phonemes_per_s > 1.2 * language_rate:
        out.append("fast?")
    return out


def main(sample_dir: Path) -> None:
    clips = read_manifest(sample_dir / "manifest.tsv")
    grouped: dict[tuple[str, str], list[Clip]] = defaultdict(list)
    for clip in clips:
        grouped[(clip.voice, clip.speaker)].append(clip)
    metrics = [speaker_metrics(group, sample_dir) for group in grouped.values()]

    rates: dict[str, list[float]] = defaultdict(list)
    for m in metrics:
        rates[m.espeak.split("-")[0]].append(m.phonemes_per_s)
    language_rate = {lang: float(np.median(r)) for lang, r in rates.items()}

    t = thresholds(metrics)
    rows = [
        {**asdict(m), "flags": "; ".join(flags(m, language_rate[m.espeak.split("-")[0]], t))}
        for m in metrics
    ]
    with (sample_dir.parent / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    items = [
        {
            **row,
            "files": [c.file for c in grouped[(row["voice"], row["speaker"])]],
            "text": "",
        }
        for row in sorted(rows, key=lambda r: (r["espeak"], r["voice"], r["speaker"]))
    ]
    template = (Path(__file__).parent / "rate.template.html").read_text()
    seeds = json.loads((Path(__file__).parent / "seed_ratings.json").read_text())
    page = template.replace("/*ITEMS*/[]", json.dumps(items)).replace("/*SEEDS*/{}", json.dumps(seeds))
    (sample_dir.parent / "rate.html").write_text(page)
    print(f"{len(rows)} speakers -> metrics.csv, rate.html")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "samples"))
