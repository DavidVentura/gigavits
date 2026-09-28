"""Phoneme-symbol coverage and greedy language selection over work/phonemized.jsonl.

Writes CSVs next to this script; prints a summary to stdout.
"""
import csv
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum

import languages as L

THRESHOLDS = (0.99, 0.999, 1.0)


class Category(Enum):
    PHONEME = "phoneme"
    SUPRA_DIACRITIC = "supra_diacritic"
    PUNCT_SPACE = "punct_space"
    TONE_DIGIT = "tone_digit"
    SPECIAL = "special"


def categorize(symbol: str) -> Category:
    if symbol in "^$_":
        return Category.SPECIAL
    if symbol.isdigit():
        return Category.TONE_DIGIT
    ucat = unicodedata.category(symbol)
    if ucat in ("Mn", "Lm") or symbol in "↑↓":
        return Category.SUPRA_DIACRITIC
    if ucat[0] in "PZS":
        return Category.PUNCT_SPACE
    assert ucat[0] == "L", (symbol, ucat)
    return Category.PHONEME


# espeak spells tones as digit strings and its IPA output turns the digit 3 into `ɜ`
# (e.g. cmn tone 2 "35" comes out as `ɜ`), so in these runs `ɜ` is a tone mark, not a vowel.
TONE_EPSILON_RUNS = {"cmn", "vi", "yue", "th"}


def categorize_in_run(run: str, symbol: str) -> Category:
    if symbol == "ɜ" and run in TONE_EPSILON_RUNS:
        return Category.TONE_DIGIT
    return categorize(symbol)


@dataclass
class RunStats:
    run: L.Run
    sentences: int
    counts: Counter
    dropped: Counter
    bigrams: Counter
    inword_markers: Counter

    def dist(self, category: Category) -> dict:
        sel = {s: c for s, c in self.counts.items() if categorize_in_run(self.run.voice, s) == category}
        total = sum(sel.values())
        return {s: c / total for s, c in sel.items()}

    def bigram_dist(self) -> dict:
        total = sum(self.bigrams.values())
        return {b: c / total for b, c in self.bigrams.items()}


def phoneme_bigrams(run: str, tokens: list) -> Counter:
    """Adjacent phoneme pairs within a word; stress/length/diacritics/tone digits are transparent,
    spaces and punctuation break the word."""
    out = Counter()
    prev = None
    for t in tokens:
        cat = categorize_in_run(run, t)
        if cat == Category.PHONEME:
            if prev is not None:
                out[(prev, t)] += 1
            prev = t
        elif cat in (Category.PUNCT_SPACE, Category.SPECIAL):
            prev = None
    return out


def inword_punctuation(phonemes: str, reference: dict) -> Counter:
    """Punctuation characters used as phonetic markers inside a word (e.g. cmn `s.` for retroflex)."""
    out = Counter()
    for i, ch in enumerate(phonemes):
        if ch not in reference or categorize(ch) != Category.PUNCT_SPACE or ch.isspace():
            continue
        if 0 < i < len(phonemes) - 1 and not phonemes[i - 1].isspace() and not phonemes[i + 1].isspace():
            out[ch] += 1
    return out


def load_stats(reference: dict) -> dict:
    assert all(len(k) == 1 for k in reference), "tokenizer replay assumes single-char symbols"
    runs = {r.voice: r for r in L.runs()}
    acc = defaultdict(lambda: dict(n=0, counts=Counter(), dropped=Counter(), bigrams=Counter(), inword=Counter()))
    for line in open(L.WORK / "phonemized.jsonl"):
        rec = json.loads(line)
        phonemes, tokens = rec["phonemes"], rec["tokens"]
        # Replays piper.rs phoneme_string_to_tokens (greedy longest match, unknown chars skipped).
        assert [c for c in phonemes if c in reference] == tokens, rec
        a = acc[rec["run"]]
        a["n"] += 1
        a["counts"].update(tokens)
        a["dropped"].update(c for c in phonemes if c not in reference)
        a["bigrams"].update(phoneme_bigrams(rec["run"], tokens))
        a["inword"].update(inword_punctuation(phonemes, reference))
    return {
        v: RunStats(runs[v], a["n"], a["counts"], a["dropped"], a["bigrams"], a["inword"])
        for v, a in acc.items()
    }


@dataclass(frozen=True)
class Step:
    k: int
    pick: str
    new_types: int
    type_cov: float
    mass_cov: float
    bigram_type_cov: float
    bigram_mass_cov: float
    supra_type_cov: float
    supra_mass_cov: float


class Universe:
    def __init__(self, stats: list):
        self.names = [s.run.voice for s in stats]
        self.types = {s.run.voice: set(s.dist(Category.PHONEME)) for s in stats}
        self.mass = self._pooled({s.run.voice: s.dist(Category.PHONEME) for s in stats})
        self.bigram_types = {s.run.voice: set(s.bigrams) for s in stats}
        self.bigram_mass = self._pooled({s.run.voice: s.bigram_dist() for s in stats})
        self.supra_types = {s.run.voice: set(s.dist(Category.SUPRA_DIACRITIC)) for s in stats}
        self.supra_mass = self._pooled({s.run.voice: s.dist(Category.SUPRA_DIACRITIC) for s in stats})

    @staticmethod
    def _pooled(dists: dict) -> dict:
        """Each language weighted equally: symbol mass = mean over languages of its per-language share."""
        pooled = Counter()
        for d in dists.values():
            for s, p in d.items():
                pooled[s] += p / len(dists)
        return pooled

    @staticmethod
    def _cov(covered: set, all_types: dict, mass: dict) -> tuple:
        union = set().union(*all_types.values())
        return len(covered & union) / len(union), sum(mass[s] for s in covered)

    def greedy(self, objective: str) -> list:
        covered, bcovered, scovered = set(), set(), set()
        remaining = list(self.names)
        steps = []
        while remaining:
            def gain(n):
                new = self.types[n] - covered
                new_mass = sum(self.mass[s] for s in new)
                # Once phoneme symbols are saturated, the order is driven by uncovered transition mass.
                new_bigram_mass = sum(self.bigram_mass[b] for b in self.bigram_types[n] - bcovered)
                if objective == "types":
                    return (len(new), new_mass, new_bigram_mass, n)
                return (new_mass, len(new), new_bigram_mass, n)
            pick = max(remaining, key=gain)
            new_types = len(self.types[pick] - covered)
            remaining.remove(pick)
            covered |= self.types[pick]
            bcovered |= self.bigram_types[pick]
            scovered |= self.supra_types[pick]
            tc, mc = self._cov(covered, self.types, self.mass)
            btc, bmc = self._cov(bcovered, self.bigram_types, self.bigram_mass)
            stc, smc = self._cov(scovered, self.supra_types, self.supra_mass)
            steps.append(Step(len(steps) + 1, pick, new_types, tc, mc, btc, bmc, stc, smc))
        return steps


def k_to_reach(steps: list, attr: str, threshold: float) -> int:
    eps = 1e-12
    return next(s.k for s in steps if getattr(s, attr) >= threshold - eps)


def quality_by_run() -> dict:
    """Best quality among app Piper voices that are phonemized with each espeak voice."""
    rank = {"x_low": 0, "low": 1, "medium": 2, "high": 3}
    best = {}
    for v in L.piper_voices():
        if not v.espeak_phonemized:
            continue
        if v.espeak_voice not in best or rank[v.quality] > rank[best[v.espeak_voice]]:
            best[v.espeak_voice] = v.quality
    return best


def write_csv(name: str, header: list, rows) -> None:
    with open(L.HERE / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def cp(s: str) -> str:
    return " ".join(f"U+{ord(c):04X}" for c in s)


def main():
    reference = json.loads(L.REFERENCE_CONFIG.read_text())["phoneme_id_map"]
    voices = L.piper_voices()
    espeak_voices = [v for v in voices if v.espeak_phonemized]
    superset = set().union(*(v.phoneme_id_map for v in espeak_voices))
    stats = load_stats(reference)
    best_quality = quality_by_run()

    write_csv("symbol_table.csv", ["symbol", "codepoint", "category", "id", "in_reference", "voices_with_symbol"],
              [(s, cp(s), categorize(s).value, reference.get(s, [None])[0], s in reference,
                sum(s in v.phoneme_id_map for v in espeak_voices))
               for s in sorted(superset)])

    write_csv("voice_map_diffs.csv",
              ["pack", "language", "quality", "espeak_voice", "phoneme_type", "n_symbols", "same_as_reference",
               "extra_vs_reference", "missing_vs_reference", "id_conflicts"],
              [(v.pack, v.language, v.quality, v.espeak_voice, v.phoneme_type, len(v.phoneme_id_map),
                v.phoneme_id_map == reference,
                " ".join(sorted(s for s in v.phoneme_id_map if s not in reference)),
                " ".join(sorted(s for s in reference if s not in v.phoneme_id_map)),
                " ".join(sorted(s for s in v.phoneme_id_map if s in reference and v.phoneme_id_map[s] != reference[s])))
               for v in voices])

    order = sorted(stats.values(), key=lambda s: (s.run.kind.value, s.run.voice))
    write_csv("symbol_counts.csv", ["run", "kind", "symbol", "codepoint", "category", "count", "share_in_category"],
              [(s.run.voice, s.run.kind.value, sym, cp(sym), categorize_in_run(s.run.voice, sym).value, c,
                round(s.dist(categorize_in_run(s.run.voice, sym)).get(sym, 0.0), 6))
               for s in order for sym, c in sorted(s.counts.items(), key=lambda x: -x[1])])
    write_csv("dropped_symbols.csv", ["run", "kind", "char", "codepoint", "unicode_name", "count", "in_app_piper_superset"],
              [(s.run.voice, s.run.kind.value, ch, cp(ch), unicodedata.name(ch, "?"), c, ch in superset)
               for s in order for ch, c in sorted(s.dropped.items(), key=lambda x: -x[1])])
    write_csv("inword_markers.csv", ["run", "kind", "symbol", "count_inside_words", "sentences"],
              [(s.run.voice, s.run.kind.value, ch, c, s.sentences)
               for s in order for ch, c in sorted(s.inword_markers.items(), key=lambda x: -x[1])])
    write_csv("runs.csv",
              ["run", "kind", "app_languages", "best_espeak_piper_quality", "sentences", "phoneme_tokens",
               "phoneme_types", "supra_types", "tone_mark_tokens", "bos_eos_pad_tokens", "dropped_chars", "phoneme_bigram_types", "packs"],
              [(s.run.voice, s.run.kind.value, ";".join(s.run.app_languages), best_quality.get(s.run.voice, ""),
                s.sentences,
                sum(c for x, c in s.counts.items() if categorize_in_run(s.run.voice, x) == Category.PHONEME),
                len(s.dist(Category.PHONEME)), len(s.dist(Category.SUPRA_DIACRITIC)),
                sum(c for x, c in s.counts.items() if categorize_in_run(s.run.voice, x) == Category.TONE_DIGIT),
                s.counts["^"] + s.counts["$"] + s.counts["_"],
                sum(s.dropped.values()), len(s.bigrams), ";".join(s.run.packs))
               for s in order])

    piper = [s for s in order if s.run.kind == L.Kind.PIPER]
    medium_plus = [s for s in piper if best_quality[s.run.voice] in ("medium", "high")]
    variants = {
        "piper_all": piper,
        "piper_no_cmn": [s for s in piper if s.run.voice != "cmn"],
        "piper_medium_plus": medium_plus,
        "piper_medium_plus_no_cmn": [s for s in medium_plus if s.run.voice != "cmn"],
        "all_app_espeak": order,
    }
    summary = []
    for vname, vstats in variants.items():
        uni = Universe(vstats)
        for objective in ("types", "mass"):
            steps = uni.greedy(objective)
            write_csv(f"greedy_{vname}_{objective}.csv",
                      ["k", "pick", "weak_teacher", "new_phoneme_types", "phoneme_type_cov", "phoneme_mass_cov",
                       "bigram_type_cov", "bigram_mass_cov", "supra_type_cov", "supra_mass_cov"],
                      [(st.k, st.pick, best_quality.get(st.pick, "non-piper") not in ("medium", "high"),
                        st.new_types, round(st.type_cov, 6), round(st.mass_cov, 6), round(st.bigram_type_cov, 6),
                        round(st.bigram_mass_cov, 6), round(st.supra_type_cov, 6), round(st.supra_mass_cov, 6))
                       for st in steps])
            for attr in ("type_cov", "mass_cov", "bigram_type_cov", "bigram_mass_cov", "supra_type_cov", "supra_mass_cov"):
                summary.append((vname, objective, len(vstats), attr,
                                *(k_to_reach(steps, attr, t) for t in THRESHOLDS)))
            print(f"\n== {vname} / greedy by {objective} ({len(vstats)} runs)")
            for st in steps[:15]:
                print(f"{st.k:2d} {st.pick:12s} +{st.new_types:2d} types={st.type_cov:.4f} mass={st.mass_cov:.5f} "
                      f"bi_types={st.bigram_type_cov:.4f} bi_mass={st.bigram_mass_cov:.5f} supra_types={st.supra_type_cov:.3f}")
    write_csv("greedy_thresholds.csv",
              ["variant", "objective", "n_runs", "metric", "k_99", "k_99_9", "k_100"], summary)

    users = defaultdict(dict)
    for s in order:
        for sym, c in s.counts.items():
            if categorize_in_run(s.run.voice, sym) in (Category.PHONEME, Category.SUPRA_DIACRITIC):
                users[sym][s.run.voice] = c
    piper_names = {s.run.voice for s in piper}
    rare_rows = []
    for sym, by_run in sorted(users.items()):
        in_piper = {r: c for r, c in by_run.items() if r in piper_names}
        if len(in_piper) <= 2 or len(by_run) <= 2:
            rare_rows.append((sym, cp(sym), categorize(sym).value, len(in_piper), len(by_run),
                              " ".join(f"{r}:{c}" for r, c in sorted(by_run.items(), key=lambda x: -x[1]))))
    write_csv("rare_symbols.csv",
              ["symbol", "codepoint", "category", "n_piper_runs", "n_all_runs", "runs_with_counts"], rare_rows)
    print("\n== thresholds")
    for row in summary:
        print(*row)


if __name__ == "__main__":
    main()
