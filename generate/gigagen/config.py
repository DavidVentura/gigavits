"""Run configuration, parsed once from TOML into typed structures."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ConfigError(ValueError):
    pass


class NoisePolicy(Enum):
    # noise 0.667 / noise_w 0.8 for every voice (PLAN 2.3)
    PIPER_DEFAULTS = "piper_defaults"
    # the values in each voice's .onnx.json, as the app plays it
    VOICE_CONFIG = "voice_config"


class RatePolicy(Enum):
    # only voices slower than their language's median are sped up to it
    RAISE_SLOW = "raise_slow"
    # every voice is moved to its language's median
    MATCH_MEDIAN = "match_median"


class SourceKind(Enum):
    TSV = "tsv"
    FLORES = "flores"


@dataclass(frozen=True)
class SentenceSource:
    kind: SourceKind
    path: Path


@dataclass(frozen=True)
class Paths:
    tiers: Path
    voice_list: Path
    kokoro_list: Path
    other_list: Path
    espeak_data: Path
    teacher_rt: Path
    # upstream fp32 Kokoro graph, and the same patched so MNN converts it (gigagen/kokoro_patch.py)
    kokoro_source_onnx: Path
    kokoro_onnx: Path
    kokoro_voices: Path
    regions: Path
    flores_codes: Path
    # the app's catalogs: public URLs per bucket path (fp32 ONNX + sidecars; MNN)
    source_catalog: Path
    mnn_catalog: Path
    extra_sources: Path
    # per-espeak-voice sentence lists written by `select` (small; kept with the code so the box reuses them)
    sentences: Path
    # The voice lists hold absolute laptop paths; on another machine they are rebased onto `bucket`.
    bucket_in_lists: Path
    bucket: Path
    out: Path


@dataclass(frozen=True)
class SentenceConfig:
    sources: tuple[SentenceSource, ...]
    train: int
    held_out: int
    core: int
    min_words: int
    max_words: int
    # Vietnamese separates syllables, not words, so its limit counts syllables
    max_words_by_espeak: dict[str, int]


@dataclass(frozen=True)
class Selection:
    voices: frozenset[str]  # empty: every voice in tiers.csv
    exclude_voices: frozenset[str]
    exclude_speakers: frozenset[str]  # speaker keys, for single speakers of multi-speaker voices
    exclude_espeak: frozenset[str]


@dataclass(frozen=True)
class Hours:
    by_tier: dict[str, float]
    lineup: float
    lineup_speakers: frozenset[str]
    by_voice: dict[str, float]
    by_speaker: dict[str, float]
    # Laptop tests cap each speaker to a few seconds of audio; None on the box.
    cap_seconds: float | None


@dataclass(frozen=True)
class RenderConfig:
    noise: NoisePolicy
    rate: RatePolicy
    second_render_fraction: float
    part_hours: float
    seed: int


@dataclass(frozen=True)
class QualityConfig:
    duration_z: float
    min_symbol_count: int
    silence_db: float
    silence_frame_s: float
    max_internal_silence_s: float
    flag_drop_fraction: float


@dataclass(frozen=True)
class Config:
    paths: Paths
    sentences: SentenceConfig
    selection: Selection
    hours: Hours
    render: RenderConfig
    quality: QualityConfig
    kokoro_url: str
    shard_bytes: int
    workers: int
    threads_per_worker: int


def _path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def parse(raw: dict, config_dir: Path) -> Config:
    p = raw["paths"]
    root = _path(config_dir, p["root"])
    paths = Paths(
        **{
            k: _path(root, p[k])
            for k in (
                "tiers", "voice_list", "kokoro_list", "other_list", "espeak_data",
                "teacher_rt", "kokoro_source_onnx", "kokoro_onnx", "kokoro_voices", "regions", "flores_codes", "sentences", "source_catalog", "mnn_catalog", "extra_sources", "bucket", "out",
            )
        },
        bucket_in_lists=Path(p["bucket_in_lists"]),
    )
    s = raw["sentences"]
    sentences = SentenceConfig(
        sources=tuple(SentenceSource(SourceKind(x["kind"]), _path(root, x["path"])) for x in s["source"]),
        train=s["train"],
        held_out=s["held_out"],
        core=s["core"],
        min_words=s["min_words"],
        max_words=s["max_words"],
        max_words_by_espeak=dict(s["max_words_by_espeak"]),
    )
    sel = raw["selection"]
    selection = Selection(
        voices=frozenset(sel["voices"]),
        exclude_voices=frozenset(sel["exclude_voices"]),
        exclude_speakers=frozenset(sel["exclude_speakers"]),
        exclude_espeak=frozenset(sel["exclude_espeak"]),
    )
    h = raw["hours"]
    hours = Hours(
        by_tier=dict(h["tier"]),
        lineup=h["lineup"],
        lineup_speakers=frozenset(h["lineup_speakers"]),
        by_voice=dict(h.get("voice", {})),
        by_speaker=dict(h.get("speaker", {})),
        cap_seconds=h.get("cap_seconds"),
    )
    r = raw["render"]
    render = RenderConfig(
        noise=NoisePolicy(r["noise"]),
        rate=RatePolicy(r["rate"]),
        second_render_fraction=r["second_render_fraction"],
        part_hours=r["part_hours"],
        seed=r["seed"],
    )
    q = raw["quality"]
    quality = QualityConfig(
        duration_z=q["duration_z"],
        min_symbol_count=q["min_symbol_count"],
        silence_db=q["silence_db"],
        silence_frame_s=q["silence_frame_s"],
        max_internal_silence_s=q["max_internal_silence_s"],
        flag_drop_fraction=q["flag_drop_fraction"],
    )
    config = Config(
        paths=paths,
        sentences=sentences,
        selection=selection,
        hours=hours,
        render=render,
        quality=quality,
        kokoro_url=raw["fetch"]["kokoro_url"],
        shard_bytes=raw["package"]["shard_bytes"],
        workers=raw["run"]["workers"],
        threads_per_worker=raw["run"]["threads_per_worker"],
    )
    _validate(config)
    return config


def _validate(config: Config) -> None:
    if not 0.0 <= config.render.second_render_fraction < 1.0:
        raise ConfigError("render.second_render_fraction must be in [0, 1)")
    if config.sentences.core > config.sentences.train:
        raise ConfigError("sentences.core must not exceed sentences.train")
    if config.render.part_hours <= 0:
        raise ConfigError("render.part_hours must be positive")
    if config.workers < 1 or config.threads_per_worker < 1:
        raise ConfigError("run.workers and run.threads_per_worker must be >= 1")
    unknown_tiers = set(config.hours.by_tier) - {"good", "clean_robot", "natural_degraded"}
    if unknown_tiers:
        raise ConfigError(f"hours.tier has unknown tiers {sorted(unknown_tiers)}")


def load(path: Path) -> Config:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return parse(raw, path.resolve().parent)
