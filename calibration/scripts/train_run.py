"""Launch stock piper1-gpl VITS training with a per-step timing callback.

Usage: train_run.py --steplog steps.csv [--cudnn-benchmark] [--compile] fit <LightningCLI args>
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback

from piper.train.__main__ import VitsLightningCLI
from piper.train.vits.dataset import VitsDataModule
from piper.train.vits.lightning import VitsModel

LOSS_KEYS = ("loss_g", "loss_d", "train_mel", "train_kl", "train_dur")


class StepTimer(Callback):
    """Writes one CSV row per optimizer step: absolute start/end time, data wait, batch audio."""

    def __init__(self, steplog: Path, compile_modules: bool, profile_batches: range) -> None:
        self.steplog = steplog
        self.compile_modules = compile_modules
        self.profile_batches = profile_batches
        self.profiler = None
        self.batches = 0
        self.file = None
        self.last_end: float | None = None
        self.batch_start = 0.0
        self.nonfinite = 0
        self.first_loss: dict[str, float] = {}
        self.last_loss: dict[str, float] = {}

    def on_train_start(self, trainer, pl_module) -> None:
        if self.compile_modules:
            # Decoder and discriminators see fixed-size segments, so they compile to static shapes.
            t0 = time.perf_counter()
            pl_module.model_g.dec = torch.compile(pl_module.model_g.dec)
            pl_module.model_d = torch.compile(pl_module.model_d)
            logging.info("torch.compile wrappers set up in %.1fs", time.perf_counter() - t0)
        self.file = open(self.steplog, "w", buffering=1)
        self.file.write("step,t_start,t_end,data_wait,batch_size,audio_seconds,max_spec_frames," + ",".join(LOSS_KEYS) + ",grad_scale\n")
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if self.batches == self.profile_batches.start and len(self.profile_batches) > 0:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            )
            self.profiler.__enter__()
        self.batch_start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        torch.cuda.synchronize()
        t_end = time.time()
        wait = self.batch_start - self.last_end if self.last_end is not None else float("nan")
        self.last_end = t_end
        self.batches += 1
        if self.profiler is not None and self.batches == self.profile_batches.stop:
            self.profiler.__exit__(None, None, None)
            averages = self.profiler.key_averages()
            report = self.steplog.with_suffix(".profile.txt")
            report.write_text(
                f"profiled batches {self.profile_batches.start}..{self.profile_batches.stop}\n"
                + averages.table(sort_by="self_cpu_time_total", row_limit=30)
                + "\n"
                + averages.table(sort_by="self_cuda_time_total", row_limit=30)
            )
            self.profiler = None
        metrics = trainer.callback_metrics
        losses = {k: float(metrics[k]) if k in metrics else float("nan") for k in LOSS_KEYS}
        if not all(math.isfinite(losses[k]) for k in ("loss_g", "loss_d")):
            self.nonfinite += 1
        if not self.first_loss:
            self.first_loss = losses
        self.last_loss = losses
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        scale = scaler.get_scale() if scaler is not None else float("nan")
        audio_seconds = float(batch.audio_lengths.sum()) / pl_module.hparams.sample_rate
        self.file.write(
            f"{trainer.global_step},{self.batch_start:.4f},{t_end:.4f},{wait:.4f},{len(batch.audio_lengths)},"
            f"{audio_seconds:.2f},{int(batch.spectrogram_lengths.max())},"
            + ",".join(f"{losses[k]:.4f}" for k in LOSS_KEYS)
            + f",{scale}\n"
        )

    def on_train_end(self, trainer, pl_module) -> None:
        self.file.close()
        summary = {
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "max_memory_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "nonfinite_steps": self.nonfinite,
            "first_losses": self.first_loss,
            "last_losses": self.last_loss,
            "params_g_m": sum(p.numel() for p in pl_module.model_g.parameters()) / 1e6,
            "params_d_m": sum(p.numel() for p in pl_module.model_d.parameters()) / 1e6,
        }
        self.steplog.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steplog", type=Path, required=True)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile-at", type=int, default=0, help="profile 10 batches starting at this batch (0 = off)")
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]

    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    VitsLightningCLI(
        VitsModel,
        VitsDataModule,
        trainer_defaults={"max_epochs": -1, "callbacks": [StepTimer(args.steplog, args.compile, range(args.profile_at, args.profile_at + 10) if args.profile_at else range(0))]},
    )


if __name__ == "__main__":
    main()
