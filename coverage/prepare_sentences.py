"""Select up to N clean FLORES-200 sentences per run and write work/sentences.tsv + sentences_used.csv.

Run with work/venv/bin/python (needs pykakasi for Japanese)."""
import csv
import re
import sys

import languages as L

TARGET = 400
MIN_WORDS = 4
MIN_CHARS_SPACELESS = 12

URL = re.compile(r"https?://|www\.|\.com\b|\.org\b|@")
DIGIT = re.compile(r"\d")
ASCII_LETTER = re.compile(r"[A-Za-z]")


def flores_lines(code: str) -> list:
    return (
        (L.FLORES / "dev" / f"{code}.dev").read_text().splitlines()
        + (L.FLORES / "devtest" / f"{code}.devtest").read_text().splitlines()
    )


def keep(line: str, code: str) -> bool:
    script = code.split("_")[1]
    if DIGIT.search(line) or URL.search(line):
        return False
    # Latin-script names inside non-Latin text make espeak switch to English phonemes.
    if script != "Latn" and ASCII_LETTER.search(line):
        return False
    if script in SPACELESS:
        return len(line) >= MIN_CHARS_SPACELESS
    return len(line.split()) >= MIN_WORDS


SPACELESS = L.SPACELESS_SCRIPTS


def to_hiragana(line: str) -> str:
    # espeak-ng's ja voice only reads kana (kanji come out as the English words "Chinese letter"),
    # and it spells out long unspaced kana runs letter by letter, so words are space-separated.
    import pykakasi
    return " ".join(part["hira"] for part in pykakasi.kakasi().convert(line.replace("・", " ")))


def main():
    rows = []
    used = []
    for run in L.runs():
        lines = flores_lines(run.flores)
        kept = [l.strip() for l in lines if keep(l.strip(), run.flores)][:TARGET]
        if run.flores == "jpn_Jpan":
            kept = [to_hiragana(s) for s in kept]
        rows += [(run.voice, run.voice, s) for s in kept]
        used.append((run.voice, run.kind.value, ";".join(run.app_languages), run.flores, len(lines), len(kept)))
    with open(L.WORK / "sentences.tsv", "w") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    with open(L.HERE / "sentences_used.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "kind", "app_languages", "flores_file", "flores_lines", "sentences_used"])
        w.writerows(used)
    for u in used:
        print(*u, file=sys.stderr)


if __name__ == "__main__":
    main()
