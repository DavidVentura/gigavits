"""Speaker, language and condition IDs assigned from the data, saved next to the run for export."""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .records import Condition, Record


class IdError(ValueError):
    pass


@dataclass(frozen=True)
class IdMaps:
    speakers: tuple[str, ...]
    languages: tuple[str, ...]
    language_espeak: tuple[tuple[str, str], ...]
    # Pairs rather than a per-language table so the order (and so the JSON) is canonical.
    language_speakers: tuple[tuple[str, str], ...]
    # Every condition the code knows gets a row from the start (inference asks for clean even when
    # no clean data exists yet); conditions added to the code later are appended by extension.
    conditions: tuple[Condition, ...] = tuple(Condition)

    def sid(self, speaker: str) -> int:
        return self.speakers.index(speaker)

    def lid(self, language: str) -> int:
        return self.languages.index(language)

    def cid(self, condition: Condition) -> int:
        return self.conditions.index(condition)

    def speakers_of(self, language: str) -> tuple[str, ...]:
        return tuple(s for lang, s in self.language_speakers if lang == language)

    def to_json(self) -> dict:
        return {
            "speakers": list(self.speakers),
            "languages": list(self.languages),
            "conditions": [c.value for c in self.conditions],
            "language_espeak": dict(self.language_espeak),
            "language_speakers": {lang: list(self.speakers_of(lang)) for lang in self.languages},
        }

    @classmethod
    def from_json(cls, raw: dict) -> IdMaps:
        if set(raw) != {"speakers", "languages", "conditions", "language_espeak", "language_speakers"}:
            raise IdError(f"unexpected id map keys {sorted(raw)}")
        conditions = tuple(Condition(c) for c in raw["conditions"])
        if len(set(conditions)) != len(conditions):
            raise IdError(f"repeated conditions {raw['conditions']}")
        languages = tuple(raw["languages"])
        if set(raw["language_espeak"]) != set(languages) or set(raw["language_speakers"]) != set(languages):
            raise IdError("language tables do not cover exactly the listed languages")
        return cls(
            speakers=tuple(raw["speakers"]),
            languages=languages,
            language_espeak=tuple((lang, raw["language_espeak"][lang]) for lang in languages),
            language_speakers=tuple((lang, s) for lang in languages for s in raw["language_speakers"][lang]),
            conditions=conditions,
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> IdMaps:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


def _append_new(existing: tuple[str, ...], seen: Iterable[str]) -> tuple[str, ...]:
    out = list(existing)
    for key in seen:
        if key not in out:
            out.append(key)
    return tuple(out)


def build_id_maps(records: Iterable[Record], previous: IdMaps | None = None) -> IdMaps:
    """Append-only: IDs of a previous map keep their positions, new keys are added in data order.

    Speakers and languages are ordered by first appearance so IDs are reproducible for the same data.
    """
    records = list(records)
    base = previous or IdMaps((), (), (), (), ())
    espeak = dict(base.language_espeak)
    for r in records:
        known = espeak.setdefault(r.language, r.espeak)
        if known != r.espeak:
            raise IdError(f"language {r.language!r} uses espeak voices {known!r} and {r.espeak!r}")
    languages = _append_new(base.languages, (r.language for r in records))
    pairs = set(base.language_speakers) | {(r.language, r.speaker) for r in records}
    speakers = _append_new(base.speakers, (r.speaker for r in records))
    return IdMaps(
        speakers=speakers,
        languages=languages,
        language_espeak=tuple((lang, espeak[lang]) for lang in languages),
        language_speakers=tuple(
            (lang, s) for lang in languages for s in speakers if (lang, s) in pairs
        ),
        conditions=base.conditions + tuple(c for c in Condition if c not in base.conditions),
    )


def require_known(records: Iterable[Record], ids: IdMaps) -> None:
    """A model's embedding tables are sized from its id map; data outside it cannot be trained."""
    known_pairs = set(ids.language_speakers)
    for r in records:
        if (r.language, r.speaker) not in known_pairs:
            raise IdError(
                f"record {r.id}: speaker {r.speaker!r} in language {r.language!r} is not in the run's id map; "
                "a new speaker or language needs new embedding rows / front ends, which this code does not add to an existing model yet"
            )
