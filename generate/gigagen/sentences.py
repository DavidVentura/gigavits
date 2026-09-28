"""Sentence selection (PLAN 2.1): sources -> class filter -> held-out -> greedy coverage order."""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tokenizer import Table, TokenizeError, piper_input, symbol_ids

from . import TOKENS_DIR
from .catalog import parse_table
from .config import Config, SentenceSource, SourceKind
from .coverage import Candidate, greedy_order, held_out
from .rt import TeacherRuntime
from .textclass import Limits, dedup_key, filter_texts

NON_PHONETIC_CLASSES = frozenset({"special", "space", "punctuation"})


@dataclass(frozen=True)
class SentenceLists:
    train: tuple[str, ...]  # coverage-greedy order: the head carries the rare symbols
    held_out: tuple[str, ...]


@dataclass(frozen=True)
class Phonemized:
    text: str
    symbols: frozenset[int] | None  # None: the shared tokenizer cannot represent the espeak output


def read_tsv_source(path: Path) -> dict[str, list[str]]:
    """coverage/work/sentences.tsv: `<espeak voice>\t<app language>\t<text>`."""
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        espeak, _app_lang, text = line.split("\t")
        out.setdefault(espeak, []).append(text)
    return out


def read_flores_source(path: Path, codes: dict[str, str], espeak_voices: frozenset[str]) -> dict[str, list[str]]:
    out = {}
    for espeak in espeak_voices & codes.keys():
        code = codes[espeak]
        out[espeak] = (
            (path / "dev" / f"{code}.dev").read_text(encoding="utf-8").splitlines()
            + (path / "devtest" / f"{code}.devtest").read_text(encoding="utf-8").splitlines()
        )
    return out


def read_sources(sources: tuple[SentenceSource, ...], codes: dict[str, str], espeak_voices: frozenset[str]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {e: [] for e in espeak_voices}
    for source in sources:
        match source.kind:
            case SourceKind.TSV:
                texts = read_tsv_source(source.path)
            case SourceKind.FLORES:
                texts = read_flores_source(source.path, codes, espeak_voices)
        for espeak in espeak_voices:
            merged[espeak] += texts.get(espeak, [])
    return merged


def phonetic_ids(table: Table) -> frozenset[int]:
    return frozenset(i for key, i in table.ids.items() if table.classes[key] not in NON_PHONETIC_CLASSES)


def to_candidate(text: str, phonemes: str, espeak: str, table: Table, phonetic: frozenset[int]) -> Phonemized:
    try:
        ids = symbol_ids(piper_input(phonemes), table.language(espeak), table)
    except TokenizeError:
        return Phonemized(text, None)
    return Phonemized(text, frozenset(ids) & phonetic)


def select(
    pools: dict[str, list[Phonemized]], train: int, n_held_out: int
) -> tuple[dict[str, SentenceLists], dict[str, dict[str, int]]]:
    """Held-out sentences of every voice are removed from every voice's training pool, so a text
    shared by two espeak voices can never be trained in one and evaluated in the other."""
    held = {e: held_out([p.text for p in pool if p.symbols is not None], n_held_out) for e, pool in pools.items()}
    never_train = {dedup_key(t) for texts in held.values() for t in texts}
    lists, stats = {}, {}
    for espeak, pool in pools.items():
        usable = [p for p in pool if p.symbols is not None]
        candidates = [Candidate(p.text, p.symbols) for p in usable if dedup_key(p.text) not in never_train]
        ordered = greedy_order(candidates, train)
        lists[espeak] = SentenceLists(train=tuple(c.text for c in ordered), held_out=tuple(held[espeak]))
        stats[espeak] = {
            "untokenizable": len(pool) - len(usable),
            "held_out": len(held[espeak]),
            "train_pool": len(candidates),
            "train": len(ordered),
        }
    return lists, stats


def _phonemize_pool(espeak: str, texts: list[str], config: Config) -> list[Phonemized]:
    table = Table.load(TOKENS_DIR / "table.json")
    phonetic = phonetic_ids(table)
    log = config.paths.out / "logs" / f"select-{espeak}.log"
    with TeacherRuntime(config.paths.teacher_rt, config.paths.espeak_data, {"engine": "espeak", "espeak": espeak}, log) as rt:
        return [to_candidate(t, rt.phonemize(t).phonemes, espeak, table, phonetic) for t in texts]


def list_dir(root: Path, espeak: str) -> Path:
    return root / espeak


def run_selection(config: Config, espeak_voices: frozenset[str]) -> None:
    codes = {r["espeak"]: r["flores"] for r in parse_table(config.paths.flores_codes.read_text(encoding="utf-8"))}
    raw = read_sources(config.sentences.sources, codes, espeak_voices)
    filtered: dict[str, list[str]] = {}
    rejected: dict[str, Counter] = {}
    for espeak in sorted(espeak_voices):
        if not raw[espeak]:
            raise ValueError(f"no sentence source has text for espeak voice {espeak!r}")
        limits = Limits(config.sentences.min_words, config.sentences.max_words_by_espeak.get(espeak, config.sentences.max_words))
        filtered[espeak], rejected[espeak] = filter_texts(raw[espeak], limits)
    (config.paths.out / "logs").mkdir(parents=True, exist_ok=True)
    voices = sorted(espeak_voices)
    with ProcessPoolExecutor(config.workers) as pool:
        phonemized = dict(zip(voices, pool.map(_phonemize_pool, voices, [filtered[e] for e in voices], [config] * len(voices))))
    lists, stats = select(phonemized, config.sentences.train, config.sentences.held_out)
    for espeak in voices:
        d = list_dir(config.paths.sentences, espeak)
        d.mkdir(parents=True, exist_ok=True)
        (d / "train.txt").write_text("".join(t + "\n" for t in lists[espeak].train), encoding="utf-8")
        (d / "held_out.txt").write_text("".join(t + "\n" for t in lists[espeak].held_out), encoding="utf-8")
        summary = {
            "source_texts": len(raw[espeak]),
            "class_rejected": {r.value: n for r, n in sorted(rejected[espeak].items(), key=lambda x: x[0].value)},
            **stats[espeak],
        }
        (d / "stats.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(f"sentences {espeak}: {json.dumps(summary)}")


def load_lists(root: Path, espeak: str) -> SentenceLists:
    d = list_dir(root, espeak)
    return SentenceLists(
        train=tuple((d / "train.txt").read_text(encoding="utf-8").splitlines()),
        held_out=tuple((d / "held_out.txt").read_text(encoding="utf-8").splitlines()),
    )
