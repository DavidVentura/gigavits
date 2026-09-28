"""Per-language evaluation data (languages.tsv): which languages Whisper CER applies to, and the
VoxLingua107 label of each espeak voice.

`whisper` is an allow-list: a code only where Whisper small is reliable enough for CER to mean
something; elsewhere phoneme error is the intelligibility measure. `voxlingua` is empty where
VoxLingua107 has no such language, so language-ID confidence does not exist for it.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from repo import EVAL

HEADER = ["espeak", "whisper", "voxlingua"]


@dataclass(frozen=True)
class LanguageInfo:
    espeak: str
    whisper: str | None
    voxlingua: str | None


class UnknownLanguage(KeyError):
    pass


@dataclass(frozen=True)
class Languages:
    by_espeak: dict[str, LanguageInfo]

    def __getitem__(self, espeak: str) -> LanguageInfo:
        if espeak not in self.by_espeak:
            raise UnknownLanguage(f"{espeak!r} is not in languages.tsv")
        return self.by_espeak[espeak]


def load_languages(path: Path = EVAL / "languages.tsv") -> Languages:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != HEADER:
            raise ValueError(f"{path}: header {reader.fieldnames}, expected {HEADER}")
        infos = [LanguageInfo(r["espeak"], r["whisper"] or None, r["voxlingua"] or None) for r in reader]
    by_espeak = {i.espeak: i for i in infos}
    if len(by_espeak) != len(infos):
        raise ValueError(f"{path}: duplicate espeak codes")
    return Languages(by_espeak)
