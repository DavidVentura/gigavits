"""Language-balanced, length-bucketed batch sampling."""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


def sampling_probabilities(languages: Sequence[str], weights: Sequence[float], seconds: Sequence[float]) -> np.ndarray:
    """Per-item probability: languages get mass proportional to sqrt(hours), items within a language
    share their language's mass in proportion to their tier weight."""
    languages = np.asarray(languages)
    weights = np.asarray(weights, dtype=np.float64)
    seconds = np.asarray(seconds, dtype=np.float64)
    if not (len(languages) == len(weights) == len(seconds)) or len(languages) == 0:
        raise ValueError("languages, weights and seconds must be equally long and non-empty")
    if (weights <= 0).any() or (seconds <= 0).any():
        raise ValueError("weights and durations must be positive")
    names, index = np.unique(languages, return_inverse=True)
    hours = np.bincount(index, weights=seconds) / 3600.0
    language_mass = np.sqrt(hours) / np.sqrt(hours).sum()
    weight_per_language = np.bincount(index, weights=weights)
    return language_mass[index] * weights / weight_per_language[index]


def plan_epoch(
    probabilities: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    num_batches: int,
    bucket_batches: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Draw num_batches * batch_size items with replacement, sort each group of bucket_batches
    batches by length so a batch holds similar lengths, then shuffle the batch order."""
    drawn = rng.choice(len(probabilities), size=num_batches * batch_size, replace=True, p=probabilities)
    group = bucket_batches * batch_size
    batches = []
    for start in range(0, len(drawn), group):
        chunk = drawn[start:start + group]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches.extend(chunk[i:i + batch_size].tolist() for i in range(0, len(chunk), batch_size))
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


class BucketedBatchSampler(Sampler[list[int]]):
    """Epoch e's batches depend only on (seed, e), so a run resumed at an epoch boundary sees the
    same batches as an uninterrupted one. Under DDP every rank plans the same epoch and takes
    every world_size-th batch."""

    def __init__(
        self,
        probabilities: np.ndarray,
        lengths: np.ndarray,
        batch_size: int,
        batches_per_epoch: int,
        bucket_batches: int,
        seed: int,
        rank: int,
        world_size: int,
    ):
        super().__init__()
        self.probabilities = probabilities
        self.lengths = lengths
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.bucket_batches = bucket_batches
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        batches = plan_epoch(
            self.probabilities,
            self.lengths,
            self.batch_size,
            self.batches_per_epoch * self.world_size,
            self.bucket_batches,
            rng,
        )
        return iter(batches[self.rank::self.world_size])
