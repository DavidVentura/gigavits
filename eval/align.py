"""Levenshtein alignment with a backtrace, and the character error rate built on it."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Sequence


class Op(Enum):
    MATCH = "match"
    SUBSTITUTE = "substitute"
    DELETE = "delete"
    INSERT = "insert"


@dataclass(frozen=True)
class Step:
    op: Op
    ref: int | None
    hyp: int | None


@dataclass(frozen=True)
class EditCount:
    edits: int
    length: int

    @property
    def rate(self) -> float:
        if self.length == 0:
            raise ZeroDivisionError("error rate of an empty reference")
        return self.edits / self.length

    def __add__(self, other: EditCount) -> EditCount:
        return EditCount(self.edits + other.edits, self.length + other.length)


def align(ref: Sequence[Hashable], hyp: Sequence[Hashable]) -> list[Step]:
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            cost[i][j] = min(diagonal, cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    steps: list[Step] = []
    i, j = n, m
    # Backtrace preference (diagonal, delete, insert) makes equal-cost alignments deterministic.
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            op = Op.MATCH if ref[i - 1] == hyp[j - 1] else Op.SUBSTITUTE
            steps.append(Step(op, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            steps.append(Step(Op.DELETE, i - 1, None))
            i -= 1
        else:
            steps.append(Step(Op.INSERT, None, j - 1))
            j -= 1
    steps.reverse()
    return steps


def edit_count(ref: Sequence[Hashable], hyp: Sequence[Hashable]) -> EditCount:
    return EditCount(sum(s.op is not Op.MATCH for s in align(ref, hyp)), len(ref))


def normalize_text(text: str) -> str:
    """Case, punctuation and symbols removed, whitespace collapsed, for comparing text with ASR output."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    kept = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in folded)
    return " ".join(kept.split())


def character_errors(reference_text: str, transcript: str) -> EditCount:
    return edit_count(normalize_text(reference_text), normalize_text(transcript))
