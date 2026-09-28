"""Evaluate per-speaker quality scores against human pass/fail verdicts.

Usage:
    venv/bin/python evaluate.py scores.csv ../voices/verdicts.csv evaluation.csv disagreements.csv
"""

import argparse
import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

METRICS = ("SIG", "BAK", "OVRL", "P808", "UTMOS", "UTMOS_min")
TARGETS = ("human", "audio")
EXTREME_FRACTION = 0.15
BOOTSTRAP_REPS = 2000
MIN_LANG_VOICES = 3


class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class Voice:
    lang: str
    voice: str
    speaker: str
    verdicts: dict[str, Verdict]
    scores: dict[str, float]


def load(scores_path: Path, verdicts_path: Path) -> list[Voice]:
    with scores_path.open(newline="") as f:
        scores = {(r["voice"], r["speaker"]): r for r in csv.DictReader(f)}
    with verdicts_path.open(newline="") as f:
        verdicts = list(csv.DictReader(f))
    missing = [(v["voice"], v["speaker"]) for v in verdicts if (v["voice"], v["speaker"]) not in scores]
    if missing:
        raise ValueError(f"verdicts without scores: {missing}")
    voices = [
        Voice(
            lang=v["lang"],
            voice=v["voice"],
            speaker=v["speaker"],
            verdicts={t: Verdict(v[t]) for t in TARGETS},
            scores={m: float(scores[(v["voice"], v["speaker"])][m]) for m in METRICS},
        )
        for v in verdicts
    ]
    return with_language_z(voices)


def with_language_z(voices: list[Voice]) -> list[Voice]:
    """Adds `<metric>_langz`: the metric standardized within its language (languages with one voice get 0)."""
    by_lang: dict[str, list[Voice]] = {}
    for v in voices:
        by_lang.setdefault(v.lang, []).append(v)
    out = []
    for v in voices:
        extra = {}
        for m in METRICS:
            values = np.array([o.scores[m] for o in by_lang[v.lang]])
            sd = values.std()
            extra[f"{m}_langz"] = 0.0 if sd == 0 else float((v.scores[m] - values.mean()) / sd)
        out.append(Voice(v.lang, v.voice, v.speaker, v.verdicts, v.scores | extra))
    return out


def pair_counts(pos: np.ndarray, neg: np.ndarray) -> tuple[float, int]:
    """Returns (wins, pairs) where ties count half."""
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).sum() + 0.5 * (diff == 0).sum()), diff.size


def pooled_auc(groups: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, int]:
    wins, pairs = 0.0, 0
    for pos, neg in groups:
        w, p = pair_counts(pos, neg)
        wins += w
        pairs += p
    return (wins / pairs if pairs else float("nan")), pairs


def split(voices: list[Voice], metric: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array([v.scores[metric] for v in voices if v.verdicts[target] is Verdict.PASS])
    neg = np.array([v.scores[metric] for v in voices if v.verdicts[target] is Verdict.FAIL])
    return pos, neg


def language_groups(voices: list[Voice], target: str) -> list[list[Voice]]:
    by_lang: dict[str, list[Voice]] = {}
    for v in voices:
        by_lang.setdefault(v.lang, []).append(v)
    return [g for g in by_lang.values() if len({v.verdicts[target] for v in g}) == 2]


def bootstrap_ci(strata: list[list[Voice]], metric: str, target: str, rng: np.random.Generator) -> tuple[float, float]:
    """Stratified bootstrap over voices; AUC pooled over strata pairs."""
    stats = []
    for _ in range(BOOTSTRAP_REPS):
        groups = []
        for stratum in strata:
            sample = [stratum[i] for i in rng.integers(0, len(stratum), len(stratum))]
            groups.append(split(sample, metric, target))
        auc, pairs = pooled_auc(groups)
        if pairs:
            stats.append(auc)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def extremes(voices: list[Voice], metric: str, target: str) -> dict[str, float]:
    ranked = sorted(voices, key=lambda v: v.scores[metric])
    k = max(1, round(EXTREME_FRACTION * len(ranked)))
    top, bottom = ranked[-k:], ranked[:k]
    return {
        "top_pass_rate": np.mean([v.verdicts[target] is Verdict.PASS for v in top]),
        "bottom_fail_rate": np.mean([v.verdicts[target] is Verdict.FAIL for v in bottom]),
        "extreme_n": k,
    }


def evaluate(voices: list[Voice], rng: np.random.Generator) -> list[dict[str, object]]:
    rows = []
    metrics = [*METRICS, *(f"{m}_langz" for m in METRICS)]
    # Noisy recordings mostly passed "human", so also test "human" with audio failures removed.
    audio_clean = [v for v in voices if v.verdicts["audio"] is Verdict.PASS]
    for subset, target, pool in (
        ("all", "human", voices),
        ("audio_pass", "human", audio_clean),
        ("all", "audio", voices),
    ):
        base_pass = np.mean([v.verdicts[target] is Verdict.PASS for v in pool])
        lang_groups = language_groups(pool, target)
        for metric in metrics:
            g_auc, g_pairs = pooled_auc([split(pool, metric, target)])
            g_lo, g_hi = bootstrap_ci([pool], metric, target, rng)
            w_auc, w_pairs = pooled_auc([split(g, metric, target) for g in lang_groups])
            w_lo, w_hi = bootstrap_ci(lang_groups, metric, target, rng)
            ext = extremes(pool, metric, target)
            rows.append({
                "target": target,
                "subset": subset,
                "metric": metric,
                "global_auc": round(g_auc, 3),
                "global_ci_lo": round(g_lo, 3),
                "global_ci_hi": round(g_hi, 3),
                "global_pairs": g_pairs,
                "within_lang_auc": round(w_auc, 3),
                "within_ci_lo": round(w_lo, 3),
                "within_ci_hi": round(w_hi, 3),
                "within_pairs": w_pairs,
                "within_langs": len(lang_groups),
                "base_pass_rate": round(float(base_pass), 3),
                "top15_pass_rate": round(float(ext["top_pass_rate"]), 3),
                "bottom15_fail_rate": round(float(ext["bottom_fail_rate"]), 3),
                "extreme_n": ext["extreme_n"],
            })
    return rows


def disagreements(voices: list[Voice], target: str, metric: str, n: int) -> list[dict[str, object]]:
    """Fail voices the metric ranks highest and pass voices it ranks lowest within their language.

    Only languages with at least MIN_LANG_VOICES voices are considered, since a z-score within a
    smaller language says nothing, and raw scores are dominated by per-language offsets.
    """
    key = f"{metric}_langz"
    lang_sizes: dict[str, int] = {}
    for v in voices:
        lang_sizes[v.lang] = lang_sizes.get(v.lang, 0) + 1
    ranked = sorted(voices, key=lambda v: v.scores[metric])
    global_pct = {id(v): i / (len(ranked) - 1) for i, v in enumerate(ranked)}
    rankable = [v for v in voices if lang_sizes[v.lang] >= MIN_LANG_VOICES]
    fails = sorted((v for v in rankable if v.verdicts[target] is Verdict.FAIL), key=lambda v: -v.scores[key])
    passes = sorted((v for v in rankable if v.verdicts[target] is Verdict.PASS), key=lambda v: v.scores[key])
    rows = []
    for kind, group in (("model_high_verdict_fail", fails[:n]), ("model_low_verdict_pass", passes[:n])):
        for v in group:
            rows.append({
                "target": target,
                "kind": kind,
                "metric": metric,
                "lang": v.lang,
                "voice": v.voice,
                "speaker": v.speaker,
                "human": v.verdicts["human"].value,
                "audio": v.verdicts["audio"].value,
                "score": round(v.scores[metric], 3),
                "global_percentile": round(global_pct[id(v)], 2),
                "lang_z": round(v.scores[key], 2),
                "lang_voices": lang_sizes[v.lang],
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", type=Path)
    parser.add_argument("verdicts", type=Path)
    parser.add_argument("evaluation_out", type=Path)
    parser.add_argument("disagreements_out", type=Path)
    parser.add_argument("--human-metric", default="UTMOS")
    parser.add_argument("--audio-metric", default="UTMOS_min")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    voices = load(args.scores, args.verdicts)
    rows = evaluate(voices, np.random.default_rng(0))
    write_csv(args.evaluation_out, rows)
    write_csv(
        args.disagreements_out,
        disagreements(voices, "human", args.human_metric, args.n) + disagreements(voices, "audio", args.audio_metric, args.n),
    )
    for r in rows:
        print(
            f"{r['target']:5} {r['subset']:10} {r['metric']:15} global {r['global_auc']:.3f} [{r['global_ci_lo']:.2f},{r['global_ci_hi']:.2f}]"
            f"  within {r['within_lang_auc']:.3f} [{r['within_ci_lo']:.2f},{r['within_ci_hi']:.2f}] ({r['within_pairs']} pairs)"
            f"  top15 pass {r['top15_pass_rate']:.2f}  bottom15 fail {r['bottom15_fail_rate']:.2f}  (base pass {r['base_pass_rate']:.2f})"
        )


if __name__ == "__main__":
    main()
