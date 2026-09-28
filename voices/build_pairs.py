"""Pairwise 'which sounds more human' page over the rendered samples.

Same-language pairs only, on the same sentence, so only the voice differs. Languages with a single
voice have nothing to compare against and are left to the per-voice page. A share of pairs is repeated with sides swapped to measure the listener's own consistency.
"""
import csv
import itertools
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPEAT_SHARE = 0.1
# Tempo is measured automatically and corrected at generation; its short pass only checks the metric.
PASSES = {
    "audio": "Which has the cleaner recording (microphone, noise, muffled or cut-off highs)?",
    "human": "Which sounds more human, less robotic?",
    "pitch": "Which has more natural pitch movement, less flat?",
}


@dataclass(frozen=True)
class Sample:
    key: str
    language: str
    file: str


def samples(manifest: Path) -> list[Sample]:
    with manifest.open() as f:
        return [
            Sample(f"{r['voice']}|{r['speaker']}", r["espeak"].split("-")[0], r["file"])
            for r in csv.DictReader(f, delimiter="\t")
            if r["idx"] == "0"
        ]


def pairs(all_samples: list[Sample], rng: random.Random) -> list[dict]:
    by_language: dict[str, list[Sample]] = defaultdict(list)
    for s in all_samples:
        by_language[s.language].append(s)
    same_language = [
        (a, b, "same language")
        for group in by_language.values()
        for a, b in itertools.combinations(group, 2)
    ]
    base = [(*rng.sample([a, b], 2), kind) for a, b, kind in same_language]
    rng.shuffle(base)
    repeats = [(b, a, kind + ", repeat") for a, b, kind in rng.sample(base, int(len(base) * REPEAT_SHARE))]
    queue = base + repeats
    rng.shuffle(queue)
    return [
        {"id": i, "kind": kind, "a": a.key, "b": b.key, "fa": a.file, "fb": b.file, "lang": a.language}
        for i, (a, b, kind) in enumerate(queue)
    ]


def tempo_pairs(all_samples: list[Sample], metrics_csv: Path, rng: random.Random) -> list[dict]:
    """Pairs spread over metric differences (large, medium, near-equal) to check the rate metric."""
    with metrics_csv.open() as f:
        rate = {f"{r['voice']}|{r['speaker']}": float(r["articulation_rate"]) for r in csv.DictReader(f)}
    by_language: dict[str, list[Sample]] = defaultdict(list)
    for s in all_samples:
        by_language[s.language].append(s)
    candidates = [
        (a, b, max(rate[a.key], rate[b.key]) / min(rate[a.key], rate[b.key]))
        for group in by_language.values()
        for a, b in itertools.combinations(group, 2)
    ]
    rng.shuffle(candidates)

    def take(low: float, high: float, n: int) -> list[tuple[Sample, Sample, float]]:
        picked, languages = [], set()
        for a, b, ratio in candidates:
            if low <= ratio < high and a.language not in languages:
                picked.append((a, b, ratio))
                languages.add(a.language)
            if len(picked) == n:
                break
        return picked

    chosen = take(1.25, 9.0, 8) + take(1.10, 1.20, 6) + take(1.0, 1.04, 6)
    queue = []
    for a, b, ratio in chosen:
        a, b = rng.sample([a, b], 2)
        expect = "close" if ratio < 1.04 else ("A" if rate[a.key] > rate[b.key] else "B")
        queue.append({"kind": f"rate ratio {ratio:.2f}", "a": a.key, "b": b.key, "fa": a.file, "fb": b.file,
                      "lang": a.language, "expect": expect})
    rng.shuffle(queue)
    return [{"id": i, **p} for i, p in enumerate(queue)]


def main() -> None:
    here = Path(__file__).parent
    all_samples = samples(here / "samples" / "manifest.tsv")
    passes = {
        name: {"question": question, "pairs": pairs(all_samples, random.Random(f"pass-{name}"))}
        for name, question in PASSES.items()
    }
    passes["tempo"] = {
        "question": "Which speaks faster? (tempo only; ignore quality)",
        "pairs": tempo_pairs(all_samples, here / "metrics.csv", random.Random("pass-tempo")),
    }
    template = (here / "compare.template.html").read_text()
    (here / "compare.html").write_text(template.replace("/*PASSES*/{}", json.dumps(passes)))
    print(f"{len(PASSES)} passes x {len(passes['audio']['pairs'])} pairs + {len(passes['tempo']['pairs'])} tempo pairs -> compare.html")


if __name__ == "__main__":
    main()
