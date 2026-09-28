"""Old Piper voice input -> shared-table input, for teacher data generation.

Every old espeak voice map is a prefix of the shared table (a map with n symbols holds exactly ids
0..n-1 with the shared ids), and piper-rs feeds a voice one token per character it knows while
silently skipping characters its map lacks. So for one espeak string:

- nothing skipped: teacher and student sequences have the same length and position k of one
  corresponds to position k of the other; only the ids of context-dependent characters differ
  (e.g. hi '.' 10 -> 165, vi '2' 132 -> 169).
- characters skipped (e.g. is '#' by the 130-symbol voices, vi tone digits by the 130-symbol
  voices, lv '`' by every voice): the teacher was fed a lossy input. The student input still comes
  from the full espeak string, and teacher outputs are not position-aligned with it.

Run as a script to list, per old voice, how many corpus sentences its teacher input is lossy for:
    python3 remap.py <bucket tts dir> <phonemized.jsonl>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tokenizer import Language, Table, UnsupportedLanguage, tokenize

PAD, BOS, EOS = "_", "^", "$"


class LayoutError(ValueError):
    pass


@dataclass(frozen=True)
class Dropped:
    index: int
    char: str


@dataclass(frozen=True)
class TeacherInput:
    ids: tuple[int, ...]
    dropped: tuple[Dropped, ...]
    whitespace_split: bool


@dataclass(frozen=True)
class Remapped:
    teacher: TeacherInput
    student_ids: tuple[int, ...]

    @property
    def aligned(self) -> bool:
        return not self.teacher.dropped and not self.teacher.whitespace_split


def teacher_input(phonemes: str, phoneme_id_map: dict[str, list[int]]) -> TeacherInput:
    """Replicates piper-rs `phonemes_to_ids`, reporting what its tokenizer skips."""
    words = phonemes.split()
    whitespace_split = bool(words) and all(w in phoneme_id_map for w in words)
    if whitespace_split:
        tokens = words
        dropped: list[Dropped] = []
    else:
        max_len = max(len(k) for k in phoneme_id_map)
        tokens, dropped, i = [], [], 0
        while i < len(phonemes):
            match = next(
                (phonemes[i:i + n] for n in range(min(max_len, len(phonemes) - i), 0, -1)
                 if phonemes[i:i + n] in phoneme_id_map),
                None,
            )
            if match is None:
                dropped.append(Dropped(i, phonemes[i]))
                i += 1
                continue
            tokens.append(match)
            i += len(match)
    pad = phoneme_id_map[PAD][0]
    ids = [phoneme_id_map[BOS][0], pad]
    for t in tokens:
        ids += [phoneme_id_map[t][0], pad]
    ids.append(phoneme_id_map[EOS][0])
    return TeacherInput(tuple(ids), tuple(dropped), whitespace_split)


def remap(phonemes: str, language: Language, phoneme_id_map: dict[str, list[int]], table: Table) -> Remapped:
    teacher = teacher_input(phonemes, phoneme_id_map)
    student = tuple(tokenize(phonemes, language, table))
    result = Remapped(teacher, student)
    if result.aligned:
        assert len(teacher.ids) == len(student), (phonemes, teacher.ids, student)
    return result


def remap_ids(
    teacher_ids: list[int], language: Language, phoneme_id_map: dict[str, list[int]], table: Table
) -> list[int]:
    """Student ids for a stored teacher id sequence.

    The teacher sequence only carries what the old map kept, so this equals the student input
    only for utterances whose teacher input was not lossy; `remap` tells that per utterance.
    """
    by_id = {v[0]: k for k, v in phoneme_id_map.items()}
    pad, bos, eos = (phoneme_id_map[c][0] for c in (PAD, BOS, EOS))
    if len(teacher_ids) < 3 or len(teacher_ids) % 2 == 0:
        raise LayoutError(f"not a [bos, pad, (id, pad)*, eos] sequence: {teacher_ids}")
    if teacher_ids[0] != bos or teacher_ids[-1] != eos or any(x != pad for x in teacher_ids[1:-1:2]):
        raise LayoutError(f"not a [bos, pad, (id, pad)*, eos] sequence: {teacher_ids}")
    body = teacher_ids[2:-1:2]
    phonemes = "".join(by_id[x] for x in body)
    return tokenize(phonemes, language, table)


def _lossy_report(bucket: Path, corpus: Path, table: Table) -> None:
    by_run: dict[str, list[str]] = {}
    with corpus.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_run.setdefault(r["run"], []).append(r["phonemes"])
    print("voice\tespeak\tmap_size\tlossy_sentences\tdropped_chars")
    for path in sorted(bucket.rglob("*.onnx.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("phoneme_type", "espeak") != "espeak" or "phoneme_id_map" not in config:
            continue
        voice = config["espeak"]["voice"]
        name = path.name.removesuffix(".onnx.json")
        id_map = config["phoneme_id_map"]
        try:
            language = table.language(voice)
        except UnsupportedLanguage:
            print(f"{name}\t{voice}\t{len(id_map)}\tunsupported language\t")
            continue
        lossy, chars = 0, Counter()
        for phonemes in by_run[voice]:
            result = remap(phonemes, language, id_map, table)
            if not result.aligned:
                lossy += 1
                chars.update(d.char for d in result.teacher.dropped)
        dropped = " ".join(f"{c!r}:{n}" for c, n in chars.most_common())
        print(f"{name}\t{voice}\t{len(id_map)}\t{lossy}/{len(by_run[voice])}\t{dropped}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: remap.py <bucket tts dir> <phonemized.jsonl>")
    _lossy_report(Path(sys.argv[1]), Path(sys.argv[2]), Table.load(Path(__file__).with_name("table.json")))
