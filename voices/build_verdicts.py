"""Per-language pass/fail page: one category at a time, voices of one language side by side."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

CATEGORIES = [
    ("human", "Does it sound human enough? (y = ok or better, n = robotic)"),
    ("audio", "Is the recording clean enough? (y = ok or better, n = bad mic, noise, muffled)"),
]


def main() -> None:
    here = Path(__file__).parent
    by_language: dict[str, list[dict]] = defaultdict(list)
    # Teacher candidates (voices/candidates/<lang>/) are judged alongside the app's voices.
    manifests = [here / "samples" / "manifest.tsv", *sorted((here / "candidates").glob("*/manifest.tsv"))]
    for manifest in manifests:
        with manifest.open() as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["idx"] == "0":
                    by_language[r["espeak"].split("-")[0]].append(
                        {"key": f"{r['voice']}|{r['speaker']}", "path": str((manifest.parent / r["file"]).relative_to(here))}
                    )
    rng = random.Random(11)
    languages = [
        {"lang": lang, "voices": rng.sample(voices, len(voices))}
        for lang, voices in sorted(by_language.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]
    data = {"categories": [{"name": n, "question": q} for n, q in CATEGORIES], "languages": languages}
    template = (here / "verdict.template.html").read_text()
    (here / "verdict.html").write_text(template.replace("/*DATA*/{}", json.dumps(data)))
    print(f"{sum(len(l['voices']) for l in languages)} voices in {len(languages)} languages -> verdict.html")


if __name__ == "__main__":
    main()
