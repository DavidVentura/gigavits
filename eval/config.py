"""Scoring configuration (TOML). A metric runs only when its section is present, so the laptop
config leaves out the models that only the GPU box downloads. Relative paths are relative to the
config file.

Model sources are tagged tables:
    { hub = "<Hugging Face repo or faster-whisper size>" }   downloaded on first use
    { path = "<local directory>" }
    { random_init = "<directory with vocab.json>" }          phoneme recognizer only: a tiny untrained
                                                              wav2vec2 with the real vocabulary, to
                                                              exercise the code path without the 1.2 GB model
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Hub:
    name: str


@dataclass(frozen=True)
class Local:
    path: Path


@dataclass(frozen=True)
class RandomInit:
    vocab_dir: Path


ModelSource = Hub | Local | RandomInit


@dataclass(frozen=True)
class PiperRsPhonemizer:
    binary: Path
    espeak_data: Path
    reference_config: Path
    reference_model: Path


@dataclass(frozen=True)
class WhisperConfig:
    model: Hub | Local
    device: str
    compute_type: str
    beam_size: int


@dataclass(frozen=True)
class PhonemeConfig:
    model: ModelSource
    device: str
    pair_counts: Path | None


@dataclass(frozen=True)
class EcapaConfig:
    model: Hub | Local
    device: str
    savedir: Path


@dataclass(frozen=True)
class SpeakerConfig:
    encoder: EcapaConfig
    reference_utterances: int


@dataclass(frozen=True)
class F0Config:
    fmin_hz: float
    fmax_hz: float
    max_length_mismatch_s: float
    min_voiced_frames: int


@dataclass(frozen=True)
class QualityConfig:
    pass


@dataclass(frozen=True)
class Config:
    threads: int
    phonemizer: PiperRsPhonemizer | None
    cer: WhisperConfig | None
    per: PhonemeConfig | None
    lid: EcapaConfig | None
    speaker: SpeakerConfig | None
    f0: F0Config | None
    quality: QualityConfig | None


class _Section:
    """Reads a TOML table, failing on missing and on unused keys."""

    def __init__(self, raw: dict, name: str, base: Path):
        self.raw, self.name, self.base, self.used = raw, name, base, set()

    def get(self, key: str, kind: type):
        if key not in self.raw:
            raise ConfigError(f"[{self.name}] needs {key!r}")
        return self.optional(key, kind)

    def optional(self, key: str, kind: type):
        self.used.add(key)
        if key not in self.raw:
            return None
        value = self.raw[key]
        if kind is float and isinstance(value, int):
            value = float(value)
        if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
            raise ConfigError(f"[{self.name}] {key} must be {kind.__name__}")
        return value

    def path(self, key: str) -> Path:
        return (self.base / self.get(key, str)).resolve()

    def optional_path(self, key: str) -> Path | None:
        value = self.optional(key, str)
        return None if value is None else (self.base / value).resolve()

    def source(self, key: str, allowed: tuple[str, ...]) -> ModelSource:
        table = self.get(key, dict)
        if len(table) != 1 or next(iter(table)) not in allowed:
            raise ConfigError(f"[{self.name}] {key} must be one of {{{' | '.join(f'{a} = ...' for a in allowed)}}}")
        tag, value = next(iter(table.items()))
        if not isinstance(value, str):
            raise ConfigError(f"[{self.name}] {key}.{tag} must be a string")
        if tag == "hub":
            return Hub(value)
        if tag == "path":
            return Local((self.base / value).resolve())
        return RandomInit((self.base / value).resolve())

    def done(self) -> None:
        unused = set(self.raw) - self.used
        if unused:
            raise ConfigError(f"[{self.name}] unknown keys {sorted(unused)}")


def parse_config(raw: dict, base: Path) -> Config:
    top = _Section(raw, "top level", base)
    metrics = _Section(top.optional("metrics", dict) or {}, "metrics", base)

    def section(parent: _Section, key: str) -> _Section | None:
        value = parent.optional(key, dict)
        return None if value is None else _Section(value, f"{parent.name}.{key}" if parent is metrics else key, base)

    threads = top.get("threads", int)

    phonemizer = None
    if (s := section(top, "phonemizer")) is not None:
        phonemizer = PiperRsPhonemizer(s.path("binary"), s.path("espeak_data"), s.path("reference_config"), s.path("reference_model"))
        s.done()

    cer = None
    if (s := section(metrics, "cer")) is not None:
        cer = WhisperConfig(s.source("model", ("hub", "path")), s.get("device", str), s.get("compute_type", str), s.get("beam_size", int))
        s.done()

    per = None
    if (s := section(metrics, "per")) is not None:
        per = PhonemeConfig(s.source("model", ("hub", "path", "random_init")), s.get("device", str), s.optional_path("pair_counts"))
        s.done()

    lid = None
    if (s := section(metrics, "lid")) is not None:
        lid = EcapaConfig(s.source("model", ("hub", "path")), s.get("device", str), s.path("savedir"))
        s.done()

    speaker = None
    if (s := section(metrics, "speaker")) is not None:
        speaker = SpeakerConfig(EcapaConfig(s.source("model", ("hub", "path")), s.get("device", str), s.path("savedir")), s.get("reference_utterances", int))
        if speaker.reference_utterances < 1:
            raise ConfigError("[metrics.speaker] reference_utterances must be at least 1")
        s.done()

    f0 = None
    if (s := section(metrics, "f0")) is not None:
        f0 = F0Config(s.get("fmin_hz", float), s.get("fmax_hz", float), s.get("max_length_mismatch_s", float), s.get("min_voiced_frames", int))
        s.done()

    quality = None
    if (s := section(metrics, "quality")) is not None:
        quality = QualityConfig()
        s.done()

    metrics.done()
    top.done()
    return Config(threads, phonemizer, cer, per, lid, speaker, f0, quality)


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return parse_config(raw, path.resolve().parent)
