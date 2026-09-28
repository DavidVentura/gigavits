"""Throughput logging: shows whether a run is compute-bound or waiting on data."""
from __future__ import annotations

import time

import lightning as L
import torch


class Throughput(L.Callback):
    """Every `every` batches logs steps/s, samples/s, the share of wall time spent waiting for the
    next batch (time between the end of one step and the start of the next), and peak GPU memory.
    Validation time is excluded from the window."""

    def __init__(self, every: int):
        super().__init__()
        self.every = every
        self._reset(time.perf_counter())
        self._batch_end: float | None = None

    def _reset(self, now: float) -> None:
        self.window_start = now
        self.steps = 0
        self.samples = 0
        self.wait = 0.0
        self.excluded = 0.0

    def on_train_start(self, trainer, pl_module) -> None:
        self._reset(time.perf_counter())

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._batch_end = None

    def on_validation_start(self, trainer, pl_module) -> None:
        self._validation_start = time.perf_counter()

    def on_validation_end(self, trainer, pl_module) -> None:
        self.excluded += time.perf_counter() - self._validation_start
        self._batch_end = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        now = time.perf_counter()
        if self._batch_end is not None:
            self.wait += now - self._batch_end

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.steps += 1
        self.samples += len(batch)
        if self.steps >= self.every:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed = now - self.window_start - self.excluded
            metrics = {
                "perf/steps_per_s": self.steps / elapsed,
                "perf/samples_per_s": self.samples / elapsed,
                "perf/data_wait_share": self.wait / elapsed,
            }
            if torch.cuda.is_available():
                metrics["perf/gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                torch.cuda.reset_peak_memory_stats()
            pl_module.log_dict(metrics, batch_size=len(batch))
            self._reset(now)
        self._batch_end = time.perf_counter()
