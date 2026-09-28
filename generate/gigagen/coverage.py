"""Held-out choice and greedy phoneme-coverage ordering of candidate sentences (PLAN 2.1)."""
from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from dataclasses import dataclass

from .textclass import dedup_key


@dataclass(frozen=True)
class Candidate:
    text: str
    symbols: frozenset[int]  # phonetic shared-table IDs occurring in the sentence


def text_hash(text: str) -> str:
    return hashlib.sha1(dedup_key(text).encode("utf-8")).hexdigest()


def held_out(texts: list[str], n: int) -> list[str]:
    """The n texts with the smallest hash: a fixed pseudo-random draw that picks the same sentences
    for every espeak voice reading the same source text (es and es-419 share FLORES Spanish)."""
    return sorted(texts, key=text_hash)[:n]


def _gain(symbols: frozenset[int], counts: Counter[int]) -> float:
    return sum(1.0 / (1 + counts[s]) for s in symbols)


def greedy_order(candidates: list[Candidate], n: int) -> list[Candidate]:
    """Repeatedly takes the sentence whose symbols are rarest in what was already taken.

    A symbol's worth is 1 / (1 + times already covered), summed over the sentence's distinct
    symbols. Worth only falls as counts rise, so stale heap scores are upper bounds and the lazy
    re-evaluation below returns exactly the plain greedy choice.
    """
    counts: Counter[int] = Counter()
    heap = [(-_gain(c.symbols, counts), text_hash(c.text), i) for i, c in enumerate(candidates)]
    heapq.heapify(heap)
    chosen: list[Candidate] = []
    while heap and len(chosen) < n:
        _, tie, i = heapq.heappop(heap)
        entry = (-_gain(candidates[i].symbols, counts), tie, i)
        if heap and entry > heap[0]:
            heapq.heappush(heap, entry)
            continue
        chosen.append(candidates[i])
        counts.update(candidates[i].symbols)
    return chosen
