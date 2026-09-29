"""Front-end init voice per language for `gigatrain init`.

    python choose_voices.py <plan.txt> <family.tsv> <ids.json> <bucket>

Per language: the good-tier lessac-family Piper voice (clean first, then highest family cosine);
else a lessac-family voice of the same language from another tier; else the closest language's
choice from CLOSEST. Prints one `--voice LANG=path` per line and the reasoning to stderr.
"""
import json
import sys
from pathlib import Path

FAMILY_MIN_COSINE = 0.3

# Languages without any lessac-family voice: nearest language with one.
CLOSEST = {
    "az": "tr", "ug": "tr", "ko": "tr",
    "bs": "sr", "hr": "sr", "sl": "sr",
    "el": "es", "eu": "es", "it": "es",
    "es-AR": "es-419",
    "et": "fi",
    "gu": "hi", "mr": "hi",
    "kn": "te", "ta": "ml",
    "he": "ar",
    "is": "nb", "sv": "nb",
    "lt": "lv",
    "ms": "id",
    "uk": "ru",
}


def main() -> None:
    plan, family_tsv, ids_json, bucket = sys.argv[1:5]
    family = {}
    for line in Path(family_tsv).read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 3:
            try:
                family[p[0]] = float(p[2])
            except ValueError:
                pass
    rows = {}
    for line in Path(plan).read_text().splitlines():
        if line.startswith("#"):
            continue
        lang, key, tier, condition, source = line.split("\t")[:5]
        if source != "PiperSource":
            continue
        voice = key.split(".")[0]
        if family.get(voice, 0.0) >= FAMILY_MIN_COSINE:
            rows.setdefault(lang, []).append((tier != "good", condition != "clean", -family[voice], voice))
    languages = json.loads(Path(ids_json).read_text())["languages"]
    chosen = {lang: sorted(rows[lang])[0][3] for lang in languages if lang in rows}
    for lang in languages:
        if lang in chosen:
            print(f"{lang}: {chosen[lang]} (family {family[chosen[lang]]:.2f})", file=sys.stderr)
            continue
        near = CLOSEST[lang]
        chosen[lang] = chosen[near] if near in chosen else sorted(rows[near])[0][3]
        print(f"{lang}: {chosen[lang]} (from {near})", file=sys.stderr)
    onnx = {p.stem: p for p in Path(bucket).rglob("*.onnx")}
    for lang in languages:
        print(f"--voice {lang}={onnx[chosen[lang]]}")


if __name__ == "__main__":
    main()
