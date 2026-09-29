"""Phoneme IDs of each run language's held-out sentences, exactly as generate tokenizes teacher input.

    generate/.venv/bin/python held_out_ids.py <generate_config.toml> <ids.json> <per_language> <out.jsonl>

Run from generate/. One JSON line per sentence: language, espeak, text, espeak_phonemes, phoneme_ids.
"""
import json
import sys
from pathlib import Path

from gigagen import TOKENS_DIR  # noqa: F401  (puts tokens/ on the path)
from gigagen.config import load
from gigagen.rt import TeacherRuntime
from gigagen.sentences import load_lists
from tokenizer import Table, piper_input, tokenize


def main() -> None:
    config = load(Path(sys.argv[1]))
    ids = json.loads(Path(sys.argv[2]).read_text())
    per_language = int(sys.argv[3])
    table = Table.load(TOKENS_DIR / "table.json")
    log = Path(sys.argv[4]).with_suffix(".teacher-rt.log")
    lines = []
    for language, espeak in sorted(ids["language_espeak"].items()):
        sentences = load_lists(config.paths.sentences, espeak).held_out[:per_language]
        with TeacherRuntime(config.paths.teacher_rt, config.paths.espeak_data, {"engine": "espeak", "espeak": espeak}, log) as rt:
            for text in sentences:
                phonemes = rt.phonemize(text).phonemes
                ids_ = tokenize(piper_input(phonemes), table.language(espeak), table)
                lines.append(json.dumps({"language": language, "espeak": espeak, "text": text,
                                         "espeak_phonemes": phonemes, "phoneme_ids": list(ids_)}, ensure_ascii=False))
    Path(sys.argv[4]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} held-out sentences -> {sys.argv[4]}")


if __name__ == "__main__":
    main()
