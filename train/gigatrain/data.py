"""Shard loading (imperative shell), dataset and batching."""
from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path

import lightning as L
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from .config import AudioConfig, DataConfig
from .ids import IdMaps
from .records import Record, Vocab, parse_manifest
from .sampling import BucketedBatchSampler, sampling_plan

_LOGGER = logging.getLogger(__name__)


class AudioError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    record: Record
    num_samples: int


def _probe(path: Path, sample_rate: int) -> int:
    info = sf.info(str(path))
    if info.samplerate != sample_rate or info.channels != 1:
        raise AudioError(f"{path}: {info.samplerate} Hz x {info.channels} channels, expected {sample_rate} Hz mono")
    return info.frames


def load_items(shards: tuple[str, ...], vocab: Vocab, sample_rate: int, workers: int) -> list[Item]:
    records: list[Record] = []
    for shard in shards:
        shard_dir = Path(shard)
        lines = (shard_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        records.extend(parse_manifest(lines, shard_dir, vocab))
    ids = [r.id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("record ids repeat across shards")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        lengths = list(pool.map(lambda r: _probe(r.audio, sample_rate), records))
    return [Item(r, n) for r, n in zip(records, lengths)]


def check_durations(item: Item, hop_length: int) -> None:
    durations = item.record.durations
    if durations is not None and sum(durations) != item.num_samples // hop_length:
        raise AudioError(
            f"{item.record.id}: durations sum to {sum(durations)} frames, audio has {item.num_samples // hop_length}"
        )


def select_items(items: list[Item], data: DataConfig, audio: AudioConfig) -> list[Item]:
    """Keep items within the configured length range; teacher durations must match the audio."""
    low, high = data.min_seconds * audio.sample_rate, data.max_utterance_s * audio.sample_rate
    kept = [i for i in items if low <= i.num_samples <= high]
    dropped: dict[str, list[float]] = {}
    for item in items:
        if not low <= item.num_samples <= high:
            dropped.setdefault(item.record.language, []).append(item.num_samples / audio.sample_rate)
    for language, seconds in sorted(dropped.items()):
        _LOGGER.info(
            "%s: dropped %d items (%.2f h) outside %.1f-%.1f s",
            language, len(seconds), sum(seconds) / 3600, data.min_seconds, data.max_utterance_s,
        )
    for item in kept:
        check_durations(item, audio.hop_length)
    return kept


def split_validation(items: list[Item], count: int) -> tuple[list[Item], list[Item]]:
    """Hold out the `count` items whose id hashes lowest: stable across runs and shard orderings."""
    ranked = sorted(items, key=lambda i: hashlib.sha1(i.record.id.encode()).digest())
    held = {i.record.id for i in ranked[:count]}
    return [i for i in items if i.record.id not in held], [i for i in items if i.record.id in held]


@dataclass
class Example:
    phoneme_ids: torch.Tensor
    durations: torch.Tensor | None
    audio: torch.Tensor
    sid: int
    lid: int
    cid: int


@dataclass
class Batch:
    phoneme_ids: torch.Tensor
    phoneme_lengths: torch.Tensor
    durations: torch.Tensor
    # Row indices worked out here on the CPU, so the training step never asks the GPU which rows
    # have teacher durations or which languages are present (each such question is a sync).
    teacher_rows: torch.Tensor
    searched_rows: torch.Tensor
    language_rows: tuple[tuple[int, torch.Tensor], ...]
    audio: torch.Tensor
    audio_lengths: torch.Tensor
    sid: torch.Tensor
    lid: torch.Tensor
    cid: torch.Tensor

    def __len__(self) -> int:
        return self.sid.size(0)

    def pin_memory(self) -> Batch:
        # DataLoader pins only tensors, mappings and sequences; a dataclass must pin itself.
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        pinned = {k: v.pin_memory() for k, v in values.items() if isinstance(v, torch.Tensor)}
        rows = tuple((lid, index.pin_memory()) for lid, index in self.language_rows)
        return Batch(**(values | pinned | {"language_rows": rows}))


class ShardDataset(Dataset):
    def __init__(self, items: list[Item], ids: IdMaps):
        self.items = items
        self.sids = [ids.sid(i.record.speaker) for i in items]
        self.lids = [ids.lid(i.record.language) for i in items]
        self.cids = [ids.cid(i.record.condition) for i in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Example:
        item = self.items[index]
        audio, _ = sf.read(str(item.record.audio), dtype="float32", always_2d=False)
        if audio.ndim != 1 or len(audio) != item.num_samples:
            raise AudioError(f"{item.record.audio}: changed since it was probed")
        durations = item.record.durations
        return Example(
            phoneme_ids=torch.tensor(item.record.phoneme_ids, dtype=torch.long),
            durations=None if durations is None else torch.tensor(durations, dtype=torch.long),
            audio=torch.from_numpy(audio),
            sid=self.sids[index],
            lid=self.lids[index],
            cid=self.cids[index],
        )


def collate(examples: list[Example]) -> Batch:
    n = len(examples)
    max_x = max(e.phoneme_ids.numel() for e in examples)
    max_y = max(e.audio.numel() for e in examples)
    phoneme_ids = torch.zeros(n, max_x, dtype=torch.long)
    durations = torch.zeros(n, max_x, dtype=torch.long)
    audio = torch.zeros(n, 1, max_y)
    for i, e in enumerate(examples):
        phoneme_ids[i, : e.phoneme_ids.numel()] = e.phoneme_ids
        audio[i, 0, : e.audio.numel()] = e.audio
        if e.durations is not None:
            durations[i, : e.durations.numel()] = e.durations
    return Batch(
        phoneme_ids=phoneme_ids,
        phoneme_lengths=torch.tensor([e.phoneme_ids.numel() for e in examples]),
        durations=durations,
        teacher_rows=torch.tensor([i for i, e in enumerate(examples) if e.durations is not None], dtype=torch.long),
        searched_rows=torch.tensor([i for i, e in enumerate(examples) if e.durations is None], dtype=torch.long),
        language_rows=tuple(
            (lid, torch.tensor([i for i, e in enumerate(examples) if e.lid == lid], dtype=torch.long))
            for lid in sorted({e.lid for e in examples})
        ),
        audio=audio,
        audio_lengths=torch.tensor([e.audio.numel() for e in examples]),
        sid=torch.tensor([e.sid for e in examples]),
        lid=torch.tensor([e.lid for e in examples]),
        cid=torch.tensor([e.cid for e in examples]),
    )


def default_workers() -> int:
    return max(0, (os.cpu_count() or 1) - 2)


class ShardDataModule(L.LightningDataModule):
    def __init__(self, train: list[Item], val: list[Item], ids: IdMaps, data: DataConfig, sample_rate: int, seed: int):
        super().__init__()
        self.sample_rate = sample_rate
        if not train:
            raise ValueError("no training items")
        self.train_items = train
        self.val_items = val
        self.ids = ids
        self.data = data
        self.seed = seed
        self.workers = default_workers() if data.num_workers is None else data.num_workers

    def _loader(self, dataset: Dataset, **kwargs) -> DataLoader:
        worker_options = (
            {"persistent_workers": True, "prefetch_factor": self.data.prefetch_factor} if self.workers > 0 else {}
        )
        return DataLoader(
            dataset,
            num_workers=self.workers,
            collate_fn=collate,
            pin_memory=self.data.pin_memory and torch.cuda.is_available(),
            **worker_options,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        records = [i.record for i in self.train_items]
        sampler = BucketedBatchSampler(
            plan=sampling_plan(
                [r.language for r in records],
                [r.weight for r in records],
                [i.num_samples / self.sample_rate for i in self.train_items],
            ),
            lengths=np.array([i.num_samples for i in self.train_items]),
            batch_size=self.data.batch_size,
            batches_per_epoch=self.data.steps_per_epoch,
            bucket_batches=self.data.bucket_batches,
            languages_per_batch=self.data.languages_per_batch,
            seed=self.seed,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
        )
        return self._loader(ShardDataset(self.train_items, self.ids), batch_sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        ordered = sorted(self.val_items, key=lambda i: i.num_samples)
        return self._loader(ShardDataset(ordered, self.ids), batch_size=self.data.batch_size, shuffle=False)
