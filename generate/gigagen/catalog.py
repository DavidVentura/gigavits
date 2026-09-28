"""Speakers and their teachers, parsed from voices/tiers.csv and the voice lists.

Pure parsers take file contents; `load` is the only function here that touches the filesystem.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import Paths


class CatalogError(ValueError):
    pass


class Tier(Enum):
    GOOD = "good"
    CLEAN_ROBOT = "clean_robot"
    NATURAL_DEGRADED = "natural_degraded"
    EXCLUDED = "excluded"


class Condition(Enum):
    CLEAN = "clean"
    NARROW_BAND = "narrow_band"
    DEGRADED = "degraded"


class PhonemeType(Enum):
    ESPEAK = "espeak"
    # the voice reads characters of the text (uk_UA-ukrainian_tts)
    TEXT = "text"


class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class TierRow:
    lang_group: str
    espeak: str
    voice: str
    speaker: str
    tier: Tier
    condition: Condition
    weight: float
    articulation_rate: float | None


@dataclass(frozen=True)
class PiperVoiceConfig:
    phoneme_type: PhonemeType
    sample_rate: int
    espeak: str
    locale: str
    num_speakers: int
    speaker_id_map: dict[str, int]
    phoneme_id_map: dict[str, list[int]]
    noise_scale: float
    length_scale: float
    noise_w: float


@dataclass(frozen=True)
class PiperSource:
    config_path: Path
    onnx: Path
    mnn: Path
    config: PiperVoiceConfig


@dataclass(frozen=True)
class KokoroSource:
    voice: str


@dataclass(frozen=True)
class KokoroJaSource:
    voice: str
    dict_path: Path


@dataclass(frozen=True)
class MmsSource:
    model: Path
    tokens: Path


@dataclass(frozen=True)
class CoquiSource:
    model: Path
    config: Path


@dataclass(frozen=True)
class Mimic3Source:
    config: Path
    mnn: Path


@dataclass(frozen=True)
class GlowTtsSource:
    model: Path
    vocoder: Path
    lexicon: Path


@dataclass(frozen=True)
class CorpusSource:
    """Recorded speech (Common Voice, IndicVoices-R, ...): loaded on the box by the corpus loaders
    (PLAN 2.3c), never rendered by this generator."""


EngineSource = KokoroSource | KokoroJaSource | MmsSource | CoquiSource | Mimic3Source | GlowTtsSource
Source = PiperSource | EngineSource | CorpusSource


@dataclass(frozen=True)
class Speaker:
    key: str
    voice: str
    speaker: str | None
    sid: int | None
    lang_group: str
    espeak: str
    language: str
    tier: Tier
    condition: Condition
    weight: float
    articulation_rate: float | None
    source: Source


def parse_tiers(text: str) -> list[TierRow]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        Verdict(r["human"]), Verdict(r["audio"])
        rows.append(
            TierRow(
                lang_group=r["lang"],
                espeak=r["espeak"],
                voice=r["voice"],
                speaker=r["speaker"],
                tier=Tier(r["tier"]),
                condition=Condition(r["condition"]),
                weight=float(r["weight"]),
                articulation_rate=float(r["articulation_rate"]) if r["articulation_rate"] else None,
            )
        )
    keys = [(r.voice, r.speaker) for r in rows]
    if len(keys) != len(set(keys)):
        raise CatalogError("tiers.csv has duplicate (voice, speaker) rows")
    return rows


def parse_table(text: str) -> list[dict[str, str]]:
    """Tab-separated table with a header line; lines starting with '#' are comments."""
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def parse_regions(text: str) -> dict[str, str]:
    return {r["locale"]: r["language"] for r in parse_table(text)}


def language_id(locale: str | None, espeak: str, regions: dict[str, str]) -> str:
    if locale is not None and locale in regions:
        return regions[locale]
    return espeak


def rebase(path: str, bucket_in_lists: Path, bucket: Path) -> Path:
    p = Path(path)
    if not p.is_relative_to(bucket_in_lists):
        raise CatalogError(f"{path} is not under {bucket_in_lists}")
    return bucket / p.relative_to(bucket_in_lists)


def parse_voice_list(text: str, bucket_in_lists: Path, bucket: Path) -> dict[str, Path]:
    """voice_list.tsv: `<voice>\t<config path>` for the app's Piper voices."""
    out = {}
    for line in text.splitlines():
        if not line:
            continue
        name, config = line.split("\t")
        out[name] = rebase(config, bucket_in_lists, bucket)
    return out


def parse_kokoro_list(text: str) -> dict[str, tuple[KokoroSource, str]]:
    """kokoro_list.tsv: `kokoro-<voice>\tkokoro:<espeak>`."""
    out = {}
    for line in text.splitlines():
        if not line:
            continue
        name, spec = line.split("\t")
        engine, espeak = spec.split(":")
        if engine != "kokoro" or not name.startswith("kokoro-"):
            raise CatalogError(f"not a kokoro line: {line!r}")
        out[name] = (KokoroSource(voice=name.removeprefix("kokoro-")), espeak)
    return out


def parse_other_list(text: str, bucket_in_lists: Path, bucket: Path) -> dict[str, tuple[EngineSource, str]]:
    """other_list.tsv: `<name>\t<engine>:<espeak>:<path>[|<path>...]` (see voices/src/main.rs)."""
    out: dict[str, tuple[EngineSource, str]] = {}
    for line in text.splitlines():
        if not line:
            continue
        name, spec = line.split("\t")
        engine, espeak, paths = spec.split(":", 2)
        p = [rebase(x, bucket_in_lists, bucket) for x in paths.split("|")]
        match engine, len(p):
            case "mms", 2:
                source: EngineSource = MmsSource(model=p[0], tokens=p[1])
            case "coqui", 2:
                source = CoquiSource(model=p[0], config=p[1])
            case "glowtts", 3:
                source = GlowTtsSource(model=p[0], vocoder=p[1], lexicon=p[2])
            case "mimic3", 1:
                source = Mimic3Source(config=p[0], mnn=p[0].with_name(p[0].name.removesuffix(".onnx.json") + ".mnn"))
            case "kokoro_ja", 1:
                source = KokoroJaSource(voice=name.removeprefix("kokoro-"), dict_path=p[0])
            case _:
                raise CatalogError(f"unknown engine line: {line!r}")
        out[name] = (source, espeak)
    return out


def parse_piper_config(text: str) -> PiperVoiceConfig:
    raw = json.loads(text)
    inference = raw["inference"]
    return PiperVoiceConfig(
        phoneme_type=PhonemeType(raw.get("phoneme_type", "espeak")),
        sample_rate=int(raw["audio"]["sample_rate"]),
        espeak=raw["espeak"]["voice"],
        locale=raw["language"]["code"],
        num_speakers=int(raw["num_speakers"]),
        speaker_id_map={k: int(v) for k, v in raw.get("speaker_id_map", {}).items()},
        phoneme_id_map={k: [int(x) for x in v] for k, v in raw["phoneme_id_map"].items()},
        noise_scale=float(inference["noise_scale"]),
        length_scale=float(inference["length_scale"]),
        noise_w=float(inference["noise_w"]),
    )


def speaker_key(voice: str, speaker: str | None) -> str:
    return voice if speaker is None else f"{voice}.{speaker}"


def _piper_speaker(row: TierRow, config: PiperVoiceConfig) -> tuple[str | None, int | None]:
    if config.num_speakers > 1:
        if row.speaker not in config.speaker_id_map:
            raise CatalogError(f"{row.voice}: speaker {row.speaker!r} not in its speaker_id_map")
        return row.speaker, config.speaker_id_map[row.speaker]
    if row.speaker != "0":
        raise CatalogError(f"{row.voice} is single-speaker but tiers.csv names speaker {row.speaker!r}")
    return None, None


def build_speakers(
    rows: list[TierRow],
    piper: dict[str, PiperSource],
    engines: dict[str, tuple[EngineSource, str]],
    regions: dict[str, str],
) -> list[Speaker]:
    speakers = []
    for row in rows:
        source: Source
        if row.voice in piper:
            source = piper[row.voice]
            if source.config.espeak != row.espeak:
                raise CatalogError(f"{row.voice}: tiers espeak {row.espeak!r} != config {source.config.espeak!r}")
            name, sid = _piper_speaker(row, source.config)
            locale: str | None = source.config.locale
        elif row.voice in engines:
            source, espeak = engines[row.voice]
            if espeak != row.espeak:
                raise CatalogError(f"{row.voice}: tiers espeak {row.espeak!r} != voice list {espeak!r}")
            if row.speaker != "0":
                raise CatalogError(f"{row.voice}: engine teachers are single-speaker")
            name, sid, locale = None, None, None
        else:
            source, name, sid, locale = CorpusSource(), None, None, None
        speakers.append(
            Speaker(
                key=speaker_key(row.voice, name),
                voice=row.voice,
                speaker=name,
                sid=sid,
                lang_group=row.lang_group,
                espeak=row.espeak,
                language=language_id(locale, row.espeak, regions),
                tier=row.tier,
                condition=row.condition,
                weight=row.weight,
                articulation_rate=row.articulation_rate,
                source=source,
            )
        )
    return speakers


def load(paths: Paths) -> list[Speaker]:
    rows = parse_tiers(paths.tiers.read_text(encoding="utf-8"))
    configs = parse_voice_list(paths.voice_list.read_text(encoding="utf-8"), paths.bucket_in_lists, paths.bucket)
    piper = {}
    for name, config_path in configs.items():
        stem = config_path.name.removesuffix(".onnx.json")
        piper[name] = PiperSource(
            config_path=config_path,
            onnx=config_path.with_name(stem + ".onnx"),
            mnn=config_path.with_name(stem + ".mnn"),
            config=parse_piper_config(config_path.read_text(encoding="utf-8")),
        )
    engines: dict[str, tuple[EngineSource, str]] = dict(parse_kokoro_list(paths.kokoro_list.read_text(encoding="utf-8")))
    engines.update(parse_other_list(paths.other_list.read_text(encoding="utf-8"), paths.bucket_in_lists, paths.bucket))
    regions = parse_regions(paths.regions.read_text(encoding="utf-8"))
    return build_speakers(rows, piper, engines, regions)
