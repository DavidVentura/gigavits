"""Class filter for training sentences (PLAN 2.1): text the espeak front end reads consistently."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum


class Reject(Enum):
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    DIGIT = "digit"
    URL = "url"
    SYMBOL = "symbol"
    ABBREVIATION = "abbreviation"
    MIXED_SCRIPT = "mixed_script"
    DUPLICATE = "duplicate"


URL = re.compile(r"https?://|www\.|\.(?:com|org|net|edu|gov)\b|@", re.IGNORECASE)
# Punctuation that espeak either reads aloud by name or handles differently per language.
SPELLED_PUNCTUATION = frozenset("#%&*/\\_|~[]{}<>§¶†‡•@^`=+")
LETTER_DOT_LETTER = re.compile(r"(?<![^\W\d_])[^\W\d_]\.[^\W\d_]")
# "Dr. ", "St. ", "J. ": a short capitalized word before a period that does not end the text.
SHORT_CAPITAL_DOT = re.compile(r"(?<![^\W\d_])[^\W\d_]{1,3}\.\s")
DOT_THEN_WORD = re.compile(r"\.\s+([^\W\d_])")


@dataclass(frozen=True)
class Limits:
    min_words: int
    max_words: int


def _is_abbreviation(text: str) -> bool:
    if LETTER_DOT_LETTER.search(text):
        return True
    for m in SHORT_CAPITAL_DOT.finditer(text):
        if m.group(0)[0].isupper():
            return True
    # Only where sentences start with a capital: Georgian has case forms but never capitalizes.
    first = next((c for c in text if c.isalpha()), "")
    if first.isupper():
        for m in DOT_THEN_WORD.finditer(text):
            if m.group(1).islower():
                return True
    for word in text.split():
        letters = [c for c in word if c.isalpha()]
        if sum(c.isupper() for c in letters) >= 2 and not any(c.islower() for c in letters):
            return True
    return False


def _is_mixed_script(text: str) -> bool:
    # Latin letters inside non-Latin text make espeak switch to English phonemes mid-sentence.
    letters = [c for c in text if c.isalpha()]
    latin = [unicodedata.name(c, "").startswith("LATIN") for c in letters]
    return any(latin) and not all(latin)


def reject_reason(text: str, limits: Limits) -> Reject | None:
    words = len(text.split())
    if words < limits.min_words:
        return Reject.TOO_SHORT
    if words > limits.max_words:
        return Reject.TOO_LONG
    if any(unicodedata.category(c).startswith("N") for c in text):
        return Reject.DIGIT
    if URL.search(text):
        return Reject.URL
    if any(c in SPELLED_PUNCTUATION or unicodedata.category(c).startswith("S") for c in text):
        return Reject.SYMBOL
    if _is_abbreviation(text):
        return Reject.ABBREVIATION
    if _is_mixed_script(text):
        return Reject.MIXED_SCRIPT
    return None


def dedup_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def filter_texts(texts: list[str], limits: Limits) -> tuple[list[str], Counter[Reject]]:
    """Keeps texts in input order; the first of several duplicates survives."""
    kept, seen, rejected = [], set(), Counter()
    for raw in texts:
        text = " ".join(raw.split())
        reason = reject_reason(text, limits)
        if reason is None and dedup_key(text) in seen:
            reason = Reject.DUPLICATE
        if reason is not None:
            rejected[reason] += 1
            continue
        seen.add(dedup_key(text))
        kept.append(text)
    return kept, rejected
