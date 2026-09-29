"""Language-balanced, language-grouped, length-bucketed batch sampling."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Sampler


@dataclass(frozen=True)
class SamplingPlan:
    """Languages get mass proportional to sqrt(hours); items within a language are drawn in
    proportion to their tier weight."""

    language_mass: np.ndarray
    members: tuple[np.ndarray, ...]
    member_probabilities: tuple[np.ndarray, ...]

    def item_probabilities(self, n_items: int) -> np.ndarray:
        """Marginal per-item probability when one language is drawn per item."""
        p = np.zeros(n_items)
        for mass, members, within in zip(self.language_mass, self.members, self.member_probabilities, strict=True):
            p[members] = mass * within
        return p


def sampling_plan(languages: Sequence[str], weights: Sequence[float], seconds: Sequence[float]) -> SamplingPlan:
    languages = np.asarray(languages)
    weights = np.asarray(weights, dtype=np.float64)
    seconds = np.asarray(seconds, dtype=np.float64)
    if not (len(languages) == len(weights) == len(seconds)) or len(languages) == 0:
        raise ValueError("languages, weights and seconds must be equally long and non-empty")
    if (weights <= 0).any() or (seconds <= 0).any():
        raise ValueError("weights and durations must be positive")
    _, index = np.unique(languages, return_inverse=True)
    hours = np.bincount(index, weights=seconds) / 3600.0
    members = tuple(np.flatnonzero(index == language) for language in range(len(hours)))
    return SamplingPlan(
        language_mass=np.sqrt(hours) / np.sqrt(hours).sum(),
        members=members,
        member_probabilities=tuple(weights[m] / weights[m].sum() for m in members),
    )


def sampling_probabilities(languages: Sequence[str], weights: Sequence[float], seconds: Sequence[float]) -> np.ndarray:
    return sampling_plan(languages, weights, seconds).item_probabilities(len(languages))


def _split(total: int, parts: int) -> list[int]:
    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]


def plan_epoch(
    plan: SamplingPlan,
    lengths: np.ndarray,
    batch_size: int,
    num_batches: int,
    bucket_batches: int,
    languages_per_batch: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Each batch draws languages_per_batch distinct languages by language mass, splits its items
    evenly between them and draws each part's items by weight, so a step runs that many front ends.

    Length bucketing: within a group of bucket_batches batches, each language's items for the
    group are drawn together, sorted by length and handed out to its parts in batch order, so the
    k-th batch of a group gets short or long items from every one of its languages alike. Batch
    order is shuffled afterwards. With several languages per batch, languages are drawn without
    replacement within a batch, which flattens their frequencies slightly towards uniform.
    """
    n_languages = len(plan.language_mass)
    if not 1 <= languages_per_batch <= min(n_languages, batch_size):
        raise ValueError(f"languages_per_batch {languages_per_batch} must be within 1..{min(n_languages, batch_size)}")
    sizes = _split(batch_size, languages_per_batch)
    batches: list[list[int]] = []
    for start in range(0, num_batches, bucket_batches):
        count = min(bucket_batches, num_batches - start)
        chosen = [rng.choice(n_languages, languages_per_batch, replace=False, p=plan.language_mass) for _ in range(count)]
        group: list[list[int]] = [[] for _ in range(count)]
        for language in np.unique(np.concatenate(chosen)):
            parts = [(b, sizes[slot]) for b in range(count) for slot, lang in enumerate(chosen[b]) if lang == language]
            drawn = rng.choice(
                plan.members[language], sum(n for _, n in parts), replace=True, p=plan.member_probabilities[language]
            )
            drawn = drawn[np.argsort(lengths[drawn], kind="stable")]
            offset = 0
            for b, n in parts:
                group[b].extend(drawn[offset:offset + n].tolist())
                offset += n
        batches.extend(group)
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


class BucketedBatchSampler(Sampler[list[int]]):
    """Epoch e's batches depend only on (seed, e), so a run resumed at an epoch boundary sees the
    same batches as an uninterrupted one. Under DDP every rank plans the same epoch and takes
    every world_size-th batch."""

    def __init__(
        self,
        plan: SamplingPlan,
        lengths: np.ndarray,
        batch_size: int,
        batches_per_epoch: int,
        bucket_batches: int,
        languages_per_batch: int,
        seed: int,
        rank: int,
        world_size: int,
    ):
        super().__init__()
        self.plan = plan
        self.lengths = lengths
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.bucket_batches = bucket_batches
        self.languages_per_batch = languages_per_batch
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
            self.plan,
            self.lengths,
            self.batch_size,
            self.batches_per_epoch * self.world_size,
            self.bucket_batches,
            self.languages_per_batch,
            rng,
        )
        return iter(batches[self.rank::self.world_size])
