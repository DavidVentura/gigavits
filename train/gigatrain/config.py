"""Typed training configuration, parsed strictly from JSON at the edge."""
from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ConfigError(ValueError):
    pass


class Precision(Enum):
    BF16_MIXED = "bf16-mixed"
    FP16_MIXED = "16-mixed"
    FP32 = "32-true"


@dataclass(frozen=True)
class ModelShape:
    """Piper VITS "medium" shapes by default."""

    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "2"
    resblock_kernel_sizes: tuple[int, ...] = (3, 5, 7)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 2), (2, 6), (3, 12))
    upsample_rates: tuple[int, ...] = (8, 8, 4)
    upsample_initial_channel: int = 256
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 8)
    gin_channels: int = 512
    classifier_hidden: int = 256


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float | None = None
    segment_size: int = 8192

    @property
    def segment_frames(self) -> int:
        return self.segment_size // self.hop_length


@dataclass(frozen=True)
class DataConfig:
    shards: tuple[str, ...]
    token_table: str
    batch_size: int = 32
    num_workers: int | None = None
    prefetch_factor: int = 4
    pin_memory: bool = True
    steps_per_epoch: int = 1000
    # Batches drawn together and sorted by length before being split, so padding stays small while
    # the sampling distribution is unchanged.
    bucket_batches: int = 32
    min_seconds: float = 0.5
    max_seconds: float = 15.0
    val_items: int = 64


@dataclass(frozen=True)
class ScheduleConfig:
    warmup_steps: int = 3_000
    gan_start_step: int = 30_000
    adversarial_ramp_steps: int = 20_000
    adversarial_weight: float = 0.02


@dataclass(frozen=True)
class OptimConfig:
    learning_rate: float = 2e-4
    learning_rate_d: float = 1e-4
    betas: tuple[float, float] = (0.8, 0.99)
    betas_d: tuple[float, float] = (0.5, 0.9)
    eps: float = 1e-9
    # Learning rate at max_steps as a fraction of the initial one; 1.0 keeps it constant.
    lr_final_ratio: float = 0.1
    grad_clip: float | None = None
    c_mel: float = 45.0
    c_kl: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    data: DataConfig
    model: ModelShape = field(default_factory=ModelShape)
    audio: AudioConfig = field(default_factory=AudioConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    precision: Precision = Precision.BF16_MIXED
    seed: int = 1234
    max_steps: int = 500_000
    log_every_n_steps: int = 50
    keep_every_n_epochs: int = 25
    cudnn_benchmark: bool = True

    def __post_init__(self) -> None:
        hop = 1
        for rate in self.model.upsample_rates:
            hop *= rate
        if hop != self.audio.hop_length:
            raise ConfigError(f"upsample rates {self.model.upsample_rates} give hop {hop}, audio hop is {self.audio.hop_length}")
        if self.audio.segment_size % self.audio.hop_length:
            raise ConfigError("segment_size must be a multiple of hop_length")
        if self.data.min_seconds * self.audio.sample_rate < self.audio.segment_size:
            raise ConfigError("min_seconds must cover at least one training segment")
        if not 0.0 < self.optim.lr_final_ratio <= 1.0:
            raise ConfigError("lr_final_ratio must be in (0, 1]")


def _parse_value(tp: typing.Any, value: typing.Any, where: str) -> typing.Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        options = typing.get_args(tp)
        if value is None:
            if type(None) not in options:
                raise ConfigError(f"{where}: null not allowed")
            return None
        (inner,) = [o for o in options if o is not type(None)]
        return _parse_value(inner, value, where)
    if dataclasses.is_dataclass(tp):
        return parse_dataclass(tp, value, where)
    if isinstance(tp, type) and issubclass(tp, Enum):
        try:
            return tp(value)
        except ValueError as e:
            raise ConfigError(f"{where}: {e}") from None
    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_parse_value(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
        return tuple(_parse_value(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp in (int, str, bool):
        if type(value) is not tp:
            raise ConfigError(f"{where}: expected {tp.__name__}, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported config type {tp!r}")


def parse_dataclass(cls: type, raw: typing.Any, where: str = "config") -> typing.Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected an object, got {raw!r}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for name, f in fields.items():
        if name in raw:
            kwargs[name] = _parse_value(hints[name], raw[name], f"{where}.{name}")
            continue
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{where}: missing key {name!r}")
    return cls(**kwargs)


def to_jsonable(value: typing.Any) -> typing.Any:
    if dataclasses.is_dataclass(value):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return value


def load_config(path: Path) -> TrainConfig:
    return parse_dataclass(TrainConfig, json.loads(path.read_text(encoding="utf-8")))
