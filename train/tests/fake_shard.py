"""Synthetic shard in the generator's format: random audio, valid shared-table IDs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from gigatrain.records import Vocab

HOP = 256
SAMPLE_RATE = 22050

# (language, espeak voice, speaker, condition, weight): en-us has two speakers so the speaker
# adversary is active there; es has one, so it is skipped.
VOICES = (
    ("en-us", "en-us", "en_US-amy", "clean", 1.0),
    ("en-us", "en-us", "en_US-joe", "degraded", 0.5),
    ("es", "es", "es_ES-davefx", "narrow_band", 1.0),
)


def phoneme_ids(rng: np.random.Generator, vocab: Vocab, n_tokens: int) -> list[int]:
    ids = [vocab.bos, vocab.pad]
    for token in rng.integers(3, vocab.size, n_tokens):
        ids += [int(token), vocab.pad]
    return ids + [vocab.eos]


def make_shard(
    shard_dir: Path, vocab: Vocab, count: int = 20, seed: int = 0, voices: tuple = VOICES, prefix: str = "item"
) -> list[dict]:
    """Even items carry teacher durations (audio length = sum * hop), odd items are aligned by MAS."""
    rng = np.random.default_rng(seed)
    (shard_dir / "audio").mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(count):
        language, espeak, speaker, condition, weight = voices[i % len(voices)]
        ids = phoneme_ids(rng, vocab, int(rng.integers(4, 10)))
        frames = int(rng.integers(50, 90))
        durations = None
        if i % 2 == 0:
            cuts = np.sort(rng.choice(np.arange(1, frames), len(ids) - 1, replace=False))
            durations = np.diff(np.concatenate([[0], cuts, [frames]])).astype(int).tolist()
        samples = frames * HOP + (0 if durations is not None else int(rng.integers(0, HOP)))
        t = np.arange(samples) / SAMPLE_RATE
        audio = 0.3 * np.sin(2 * np.pi * rng.uniform(100, 300) * t) + 0.05 * rng.standard_normal(samples)
        relative = f"audio/{prefix}-{i:04d}.flac"
        sf.write(shard_dir / relative, audio.astype(np.float32), SAMPLE_RATE, format="FLAC")
        records.append({
            "id": f"{prefix}-{i:04d}",
            "audio": relative,
            "language": language,
            "espeak": espeak,
            "speaker": speaker,
            "condition": condition,
            "phoneme_ids": ids,
            "durations": durations,
            "text": f"sentence {i}",
            "teacher": "fake",
            "weight": weight,
        })
    with open(shard_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return records
