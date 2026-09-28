"""Phone units for phoneme error rate, and the split of errors by training phoneme-pair coverage.

Both sides are reduced to the same units: a base phone letter plus the marks written after it
(length, palatalization, aspiration, nasalization, dental, ...). Stress, intonation arrows, tone
tokens, tie bars and leftover legacy characters carry no segment of their own and are dropped, so
the reference (piper-rs espeak strings through the shared tokenizer) and the recognizer output
(wav2vec2-lv-60-espeak-cv-ft tokens, which are unstressed) compare on segments only.

Pairs follow the coverage analysis (coverage/analyze.py `phoneme_bigrams`): adjacent base phones
within a word, marks transparent, spaces and punctuation break the word.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from align import EditCount, Op, align


class SymbolClass(Enum):
    PHONE = "phone"
    MARK = "mark"
    BOUNDARY = "boundary"
    TRANSPARENT = "transparent"


PROSODIC_MODIFIERS = frozenset("ˈˌ↓↑")
TIE = "͡"


@dataclass(frozen=True)
class Unit:
    base: str
    marks: str

    @property
    def label(self) -> str:
        return self.base + self.marks


Word = tuple[Unit, ...]
Pair = tuple[str, str]


class MalformedPhonemes(ValueError):
    pass


def classify_table_key(key: str, table_class: str) -> SymbolClass:
    """Class of a shared-table token (tokens/table.json `class`)."""
    if table_class in ("vowel", "consonant"):
        return SymbolClass.PHONE
    if table_class == "modifier":
        return SymbolClass.TRANSPARENT if key in PROSODIC_MODIFIERS else SymbolClass.MARK
    if table_class == "diacritic":
        return SymbolClass.TRANSPARENT if key == TIE else SymbolClass.MARK
    if table_class in ("space", "punctuation"):
        return SymbolClass.BOUNDARY
    if table_class in ("tone", "legacy"):
        return SymbolClass.TRANSPARENT
    raise MalformedPhonemes(f"token {key!r} of class {table_class!r} cannot occur inside an utterance")


def classify_recognized_char(char: str, table_classes: dict[str, str]) -> SymbolClass:
    """Class of one character of recognizer output, which may hold symbols the shared table lacks."""
    if char in table_classes:
        if table_classes[char] == "special":
            return SymbolClass.TRANSPARENT
        cls = classify_table_key(char, table_classes[char])
        # The recognizer emits no word boundaries; punctuation-like characters in its vocabulary
        # are espeak notation fragments (e.g. 's.' retroflex), not boundaries.
        return SymbolClass.TRANSPARENT if cls is SymbolClass.BOUNDARY else cls
    if char.isdigit():
        return SymbolClass.TRANSPARENT
    category = unicodedata.category(char)
    if category in ("Mn", "Lm"):
        return SymbolClass.MARK
    if category[0] == "L":
        return SymbolClass.PHONE
    return SymbolClass.TRANSPARENT


def _unit(base: str, marks: list[str]) -> Unit:
    return Unit(base, "".join(sorted(marks)))


def reference_words(keys: Sequence[str], table_classes: dict[str, str]) -> list[Word]:
    """Words of units from the shared-table token keys of one utterance (tokenizer `symbol_ids`)."""
    words: list[Word] = []
    current: list[Unit] = []
    base: str | None = None
    marks: list[str] = []
    for i, key in enumerate(keys):
        cls = classify_table_key(key, table_classes[key])
        if cls is SymbolClass.MARK:
            if base is None:
                raise MalformedPhonemes(f"mark {key!r} at {i} follows no phone in {''.join(keys)!r}")
            marks.append(key)
            continue
        if base is not None and cls is not SymbolClass.TRANSPARENT:
            current.append(_unit(base, marks))
            base, marks = None, []
        if cls is SymbolClass.PHONE:
            base = key
        elif cls is SymbolClass.BOUNDARY and current:
            words.append(tuple(current))
            current = []
    if base is not None:
        current.append(_unit(base, marks))
    if current:
        words.append(tuple(current))
    return words


def recognized_units(tokens: Iterable[str], table_classes: dict[str, str]) -> list[Unit]:
    units: list[Unit] = []
    base: str | None = None
    marks: list[str] = []
    for char in "".join(tokens):
        cls = classify_recognized_char(char, table_classes)
        if cls is SymbolClass.MARK:
            # Recognizer output is unconstrained; a mark with no phone before it describes nothing.
            if base is not None:
                marks.append(char)
            continue
        if cls is SymbolClass.PHONE:
            if base is not None:
                units.append(_unit(base, marks))
            base, marks = char, []
    if base is not None:
        units.append(_unit(base, marks))
    return units


def word_pairs(word: Word) -> list[Pair]:
    return [(a.base, b.base) for a, b in zip(word, word[1:])]


def count_pairs(words: Iterable[Word]) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for word in words:
        counts.update(word_pairs(word))
    return counts


def unit_edits(reference: Sequence[Unit], recognized: Sequence[Unit]) -> list[int]:
    """Edits charged to each reference unit; an insertion is charged to the reference unit before it
    (or after it, at the start), so the per-unit edits sum to the total edit distance."""
    if not reference:
        raise MalformedPhonemes("phoneme error rate of an empty reference")
    edits = [0] * len(reference)
    last_ref: int | None = None
    pending_inserts = 0
    for step in align([u.label for u in reference], [u.label for u in recognized]):
        if step.op is Op.INSERT:
            if last_ref is None:
                pending_inserts += 1
            else:
                edits[last_ref] += 1
            continue
        last_ref = step.ref
        edits[step.ref] += (step.op is not Op.MATCH) + pending_inserts
        pending_inserts = 0
    return edits


class PairStatus(Enum):
    SEEN = "seen"
    UNSEEN = "unseen"


def unit_pair_status(words: Sequence[Word], trained: Counter[Pair]) -> list[PairStatus]:
    """Per unit (flattened over words): UNSEEN when a within-word pair it belongs to never occurred
    in training. A unit alone in its word has no pair and counts as SEEN."""
    out: list[PairStatus] = []
    for word in words:
        pairs = word_pairs(word)
        for i in range(len(word)):
            touching = pairs[max(0, i - 1):i + 1]
            unseen = any(trained[p] == 0 for p in touching)
            out.append(PairStatus.UNSEEN if unseen else PairStatus.SEEN)
    return out


def split_by_pair_status(words: Sequence[Word], edits: Sequence[int], trained: Counter[Pair]) -> dict[PairStatus, EditCount]:
    statuses = unit_pair_status(words, trained)
    if len(statuses) != len(edits):
        raise ValueError(f"{len(edits)} per-unit edits for {len(statuses)} reference units")
    return {
        status: EditCount(
            sum(e for e, s in zip(edits, statuses) if s is status),
            sum(1 for s in statuses if s is status),
        )
        for status in PairStatus
    }
