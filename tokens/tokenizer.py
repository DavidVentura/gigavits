"""Shared phoneme-ID tokenizer for espeak-ng 1.52 output as produced by piper-rs.

(Named tokenizer.py rather than tokenize.py: a local tokenize.py shadows the stdlib module that
linecache/traceback import, which breaks every traceback printed from this directory.)

Input is the string piper-rs feeds a Piper voice: `espeak_phonemize` builds, for each espeak clause,
the clause's IPA followed by the clause's terminating punctuation and one space ("<ipa><p> ", only
when the clause had a terminator), joins sentence chunks with " ", applies NFD, and
`PiperModel::synthesize_phonemes` then trims it (`piper_input` below). espeak's own IPA never
contains text punctuation, so every character is phonetic except the terminators piper-rs appends.
Tokenizing the trimmed string keeps student and teacher sequences position-aligned 1:1.

Algorithm (the Rust port implements exactly this, driven by table.json):

For each character c at index i of the NFD input, for language L:
1. rule = the first rule in table.json `rules` with `char == c`, L in `languages`, and a matching
   `context`:
     - `prev_chars` / `prev_classes`: the character before i (after skipping characters whose
       table class is in `skip_classes`) is in `prev_chars` or has a table class in
       `prev_classes`; at least one of the two must hold when either is given.
     - `next_chars`: the character after i is in `next_chars`.
2. Clause punctuation: c is in `clause_punctuation` and is the last character (the trim removed
   the space after the final terminator): punctuation. The trim makes this shape ambiguous too: a
   clause without a terminator that ends in a phonetic '.' (bn "r." before a danda espeak does not
   treat as a terminator) reads as punctuation; the terminator after a rule context (ar "a.") is
   far more common. Tokenizing espeak-rs chunks instead of the joined string removes both.
   Or c is in `clause_punctuation` and s[i+1] == ' '. Then:
     - if s[i+2] is ' ' or the end of the string, the clause ended a sentence chunk: punctuation;
     - otherwise (exactly one space, text continues), c is punctuation unless `rule` matched. This
       shape is ambiguous as well: a word-final phonetic '.' (hi-family r., ar a. i. u.) looks
       exactly like a mid-chunk clause terminator after the same letter.
   Punctuation emits the token whose key is c.
3. Else if `rule` matched, emit rule.token.
4. Else if c is in `reserved`, raise UnknownSymbol (a reserved character only ever means
   something through step 2 or 3).
5. Else if c is a token key, emit its id.
6. Else raise UnknownSymbol.

Every input character yields exactly one token, so sequences stay position-aligned with what the
old per-character Piper voices were fed (see remap.py). The model input is then
[bos, pad, t1, pad, t2, pad, ..., tn, pad, eos], as in piper-rs `phonemes_to_ids`.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path


class TokenizeError(ValueError):
    pass


class UnsupportedLanguage(TokenizeError):
    pass


class NotNfd(TokenizeError):
    pass


class UnknownSymbol(TokenizeError):
    def __init__(self, language: str, phonemes: str, index: int):
        char = phonemes[index]
        snippet = phonemes[max(0, index - 12):index + 12]
        super().__init__(
            f"{language}: no token for {char!r} (U+{ord(char):04X}) at {index} in {snippet!r}"
        )
        self.language = language
        self.char = char
        self.index = index


@dataclass(frozen=True)
class Language:
    code: str


@dataclass(frozen=True)
class Context:
    prev_chars: frozenset[str]
    prev_classes: frozenset[str]
    skip_classes: frozenset[str]
    next_chars: frozenset[str]


@dataclass(frozen=True)
class Rule:
    char: str
    context: Context
    token_id: int


@dataclass(frozen=True)
class Table:
    ids: dict[str, int]
    classes: dict[str, str]
    pad: int
    bos: int
    eos: int
    clause_punctuation: frozenset[str]
    reserved: frozenset[str]
    languages: frozenset[str]
    unsupported: dict[str, str]
    rules: dict[tuple[str, str], tuple[Rule, ...]]

    @classmethod
    def load(cls, path: Path) -> Table:
        raw = json.loads(path.read_text(encoding="utf-8"))
        ids = {t["key"]: t["id"] for t in raw["tokens"]}
        assert sorted(ids.values()) == list(range(len(ids))), "token ids must be dense and unique"
        languages = frozenset(raw["languages"])
        rules: dict[tuple[str, str], list[Rule]] = {}
        for r in raw["rules"]:
            ctx = r["context"]
            rule = Rule(
                char=r["char"],
                context=Context(
                    prev_chars=frozenset(ctx.get("prev_chars", ())),
                    prev_classes=frozenset(ctx.get("prev_classes", ())),
                    skip_classes=frozenset(ctx.get("skip_classes", ())),
                    next_chars=frozenset(ctx.get("next_chars", ())),
                ),
                token_id=ids[r["token"]],
            )
            for lang in r["languages"]:
                assert lang in languages, (lang, r)
                rules.setdefault((lang, r["char"]), []).append(rule)
        return cls(
            ids=ids,
            classes={t["key"]: t["class"] for t in raw["tokens"]},
            pad=ids[raw["pad"]],
            bos=ids[raw["bos"]],
            eos=ids[raw["eos"]],
            clause_punctuation=frozenset(raw["clause_punctuation"]),
            reserved=frozenset(raw["reserved"]),
            languages=languages,
            unsupported=dict(raw["unsupported_languages"]),
            rules={k: tuple(v) for k, v in rules.items()},
        )

    def language(self, code: str) -> Language:
        if code in self.unsupported:
            raise UnsupportedLanguage(f"{code}: {self.unsupported[code]}")
        if code not in self.languages:
            raise UnsupportedLanguage(f"{code}: not an espeak voice covered by the table")
        return Language(code)


def _context_matches(ctx: Context, phonemes: str, i: int, table: Table) -> bool:
    if ctx.prev_chars or ctx.prev_classes:
        j = i - 1
        while j >= 0 and table.classes.get(phonemes[j]) in ctx.skip_classes:
            j -= 1
        if j < 0:
            return False
        prev = phonemes[j]
        if prev not in ctx.prev_chars and table.classes.get(prev) not in ctx.prev_classes:
            return False
    if ctx.next_chars:
        if i + 1 >= len(phonemes) or phonemes[i + 1] not in ctx.next_chars:
            return False
    return True


def _matching_rule(phonemes: str, i: int, language: Language, table: Table) -> Rule | None:
    for rule in table.rules.get((language.code, phonemes[i]), ()):
        if _context_matches(rule.context, phonemes, i, table):
            return rule
    return None


def piper_input(espeak_phonemes: str) -> str:
    """What piper-rs maps to IDs: the `espeak_phonemize` string, trimmed, NFD."""
    return unicodedata.normalize("NFD", espeak_phonemes.strip())


def _is_clause_punctuation(phonemes: str, i: int, rule: Rule | None, table: Table) -> bool:
    if phonemes[i] not in table.clause_punctuation:
        return False
    if i + 1 == len(phonemes):
        return True
    if phonemes[i + 1:i + 2] != " ":
        return False
    if i + 2 == len(phonemes) or phonemes[i + 2] == " ":
        return True
    return rule is None


def symbol_ids(phonemes: str, language: Language, table: Table) -> list[int]:
    """One token id per input character, without BOS/PAD/EOS."""
    if not unicodedata.is_normalized("NFD", phonemes):
        raise NotNfd(f"{language.code}: input is not NFD: {phonemes!r}")
    out = []
    for i, c in enumerate(phonemes):
        rule = _matching_rule(phonemes, i, language, table)
        if _is_clause_punctuation(phonemes, i, rule, table):
            out.append(table.ids[c])
            continue
        if rule is not None:
            out.append(rule.token_id)
            continue
        if c in table.reserved or c not in table.ids:
            raise UnknownSymbol(language.code, phonemes, i)
        out.append(table.ids[c])
    return out


def tokenize(phonemes: str, language: Language, table: Table) -> list[int]:
    ids = [table.bos, table.pad]
    for sid in symbol_ids(phonemes, language, table):
        ids.append(sid)
        ids.append(table.pad)
    ids.append(table.eos)
    return ids
