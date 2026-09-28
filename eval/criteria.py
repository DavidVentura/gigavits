"""Evaluate PLAN.md's pass criteria against score outputs.

Usage:
    venv/bin/python criteria.py --spec criteria.toml --originals <rows.csv>... --candidate <rows.csv>...
        --tiers ../voices/tiers.csv --tag <checkpoint tag> --out <dir>
        [--verdicts <verdicts.csv>] [--ab <ab.csv>] [--ratings <ratings.csv>] [--runtime <runtime.tsv>]

--originals: score rows of the original voices (baseline seed renders: lineup originals, the best
    original voices of every language, the weak-language references).
--candidate: score rows of the new model; voice|speaker is the lineup voice, espeak the language
    it speaks. Own-language rows carry speaker similarity against the original voice, cross-language
    rows against the new model's own-language output of the same voice.
--verdicts:  the verdict page export (voices/verdict.html) after judging the manifests written here.
--ab, --ratings: exports of the listening page (listening.py).
--runtime:   variant (base | <name>) and ratio: phone CPU time per audio second relative to a
    current medium voice.

Writes <out>/criteria.csv (one line per test and scope), <out>/needs_ears.csv (what a listener
still has to judge) and <out>/verdict/<espeak>/manifest.tsv + WAVs for the verdict page builder.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import fmean
from typing import Iterable

import soundfile as sf

from languages import Languages, load_languages
from manifest import ReferenceVoice, VoiceKey, write_manifest
from scores import ScoreRow, mean_of, pooled_rate, read_rows


@dataclass(frozen=True)
class Thresholds:
    """PLAN.md "Pass criteria"; changed only with a recorded reason."""

    ab_min_sentences: int = 30
    ab_max_original_preferred: float = 0.60
    cer_margin: float = 0.02
    own_speaker_similarity: float = 0.80
    own_utmos_margin: float = 0.1
    native_rating: float = 3.5
    lid_margin: float = 0.1
    cross_speaker_similarity: float = 0.70
    weak_utmos_margin: float = 0.2
    weak_per_margin: float = 0.05
    runtime_base: float = 1.10
    runtime_variant: float = 2.0
    utmos_relisten: float = 2.9


THRESHOLDS = Thresholds()


class Status(Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_MEASURED = "not_measured"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class Result:
    test: str
    voice: str
    language: str
    measured: float | None
    reference: float | None
    threshold: str
    status: Status
    note: str = ""


@dataclass(frozen=True)
class NeedsEars:
    task: str
    voice: str
    language: str
    where: str
    why: str


@dataclass(frozen=True)
class LineupVoice:
    key: VoiceKey
    language: str


@dataclass(frozen=True)
class Spec:
    lineup: tuple[LineupVoice, ...]
    native_rating_languages: frozenset[str]
    weak_language: str
    weak_reference: VoiceKey


def parse_spec(raw: dict) -> Spec:
    expected = {"lineup", "native_rating_languages", "weak_language", "weak_reference"}
    if set(raw) != expected:
        raise ValueError(f"criteria spec keys {sorted(raw)}, expected {sorted(expected)}")
    lineup = []
    for entry in raw["lineup"]:
        if set(entry) != {"voice", "speaker", "language"}:
            raise ValueError(f"lineup entry {entry} needs exactly voice, speaker, language")
        lineup.append(LineupVoice(VoiceKey(entry["voice"], entry["speaker"]), entry["language"]))
    keys = [v.key for v in lineup]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate lineup voices")
    return Spec(tuple(lineup), frozenset(raw["native_rating_languages"]), raw["weak_language"], VoiceKey.parse(raw["weak_reference"]))


@dataclass(frozen=True)
class AbAnswer:
    listener: str
    key: VoiceKey
    item: str
    original_side: str
    answer: str

    @property
    def original_preferred(self) -> float:
        if self.answer == "tie":
            return 0.5
        return 1.0 if self.answer == self.original_side else 0.0


@dataclass(frozen=True)
class Rating:
    listener: str
    language: str
    key: VoiceKey
    item: str
    score: int
    note: str


@dataclass(frozen=True)
class Verdict:
    key: VoiceKey
    language: str
    human: str


def _read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        absent = [c for c in columns if c not in (reader.fieldnames or ())]
        if absent:
            raise ValueError(f"{path}: no column {absent}")
        return list(reader)


def parse_ab(path: Path) -> list[AbAnswer]:
    out = []
    for r in _read_csv(path, ["listener", "voice", "speaker", "item", "original_side", "answer"]):
        if r["original_side"] not in ("A", "B") or r["answer"] not in ("A", "B", "tie"):
            raise ValueError(f"{path}: bad A/B row {r}")
        out.append(AbAnswer(r["listener"], VoiceKey(r["voice"], r["speaker"]), r["item"], r["original_side"], r["answer"]))
    return out


def parse_ratings(path: Path) -> list[Rating]:
    out = []
    for r in _read_csv(path, ["listener", "language", "voice", "speaker", "item", "score", "note"]):
        score = int(r["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"{path}: score {score} outside 1-5")
        out.append(Rating(r["listener"], r["language"], VoiceKey(r["voice"], r["speaker"]), r["item"], score, r["note"]))
    return out


def parse_original_human(path: Path) -> dict[VoiceKey, str]:
    rows = _read_csv(path, ["voice", "speaker", "human"])
    return {VoiceKey(r["voice"], r["speaker"]): r["human"] for r in rows}


def original_human_by_language(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for r in _read_csv(path, ["espeak", "human"]):
        out[r["espeak"]].append(r["human"])
    return out


VERDICT_SEPARATOR = "~"


def verdict_voice_name(tag: str, key: VoiceKey, espeak: str) -> str:
    # The verdict page keys answers by voice|speaker and groups by the language's first subtag, so
    # the name must differ from the original voice's and keep es and es-419 apart.
    return VERDICT_SEPARATOR.join((tag, key.voice, espeak))


def parse_verdicts(path: Path, tag: str) -> list[Verdict]:
    out = []
    for r in _read_csv(path, ["voice", "speaker", "human"]):
        parts = r["voice"].split(VERDICT_SEPARATOR)
        if len(parts) != 3 or parts[0] != tag:
            continue
        if r["human"] not in ("pass", "fail", ""):
            raise ValueError(f"{path}: human verdict {r['human']!r}")
        if r["human"]:
            out.append(Verdict(VoiceKey(parts[1], r["speaker"]), parts[2], r["human"]))
    return out


def parse_runtime(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["variant", "ratio"]:
            raise ValueError(f"{path}: header {reader.fieldnames}, expected variant, ratio")
        return {r["variant"]: float(r["ratio"]) for r in reader}


@dataclass(frozen=True)
class Inputs:
    spec: Spec
    languages: Languages
    originals: list[ScoreRow]
    candidate: list[ScoreRow]
    original_human: dict[VoiceKey, str]
    original_human_by_language: dict[str, list[str]]
    verdicts: list[Verdict] | None
    ab: list[AbAnswer] | None
    ratings: list[Rating] | None
    runtime: dict[str, float] | None


def _select(rows: Iterable[ScoreRow], key: VoiceKey | None = None, espeak: str | None = None) -> list[ScoreRow]:
    return [r for r in rows if (key is None or r.key == key) and (espeak is None or r.espeak == espeak)]


def _common_sentences(a: list[ScoreRow], b: list[ScoreRow]) -> tuple[list[ScoreRow], list[ScoreRow]]:
    shared = {r.text for r in a} & {r.text for r in b}
    return [r for r in a if r.text in shared], [r for r in b if r.text in shared]


def _judge(value: float | None, reference: float | None, passes: bool, test: str, voice: str, language: str,
           threshold: str, note: str = "") -> Result:
    if value is None:
        return Result(test, voice, language, None, reference, threshold, Status.NOT_MEASURED, note)
    return Result(test, voice, language, value, reference, threshold, Status.PASS if passes else Status.FAIL, note)


def _check_reference(rows: list[ScoreRow], key: VoiceKey, where: str) -> None:
    for r in rows:
        if r.speaker_reference is not None and ReferenceVoice.parse(r.speaker_reference).key != key:
            raise ValueError(f"{where}: {r.path} measures speaker similarity against {r.speaker_reference}, expected {key}")


def own_language(inputs: Inputs, lineup: LineupVoice) -> list[Result]:
    t = THRESHOLDS
    name, lang = str(lineup.key), lineup.language
    cand = _select(inputs.candidate, lineup.key, lang)
    orig = _select(inputs.originals, lineup.key, lang)
    if not cand:
        raise ValueError(f"no candidate rows for lineup voice {name} in its own language {lang}")
    if not orig:
        raise ValueError(f"no original rows for lineup voice {name} in {lang}")
    results = []

    answers = [a for a in inputs.ab or [] if a.key == lineup.key]
    if inputs.ab is None or not answers:
        results.append(Result("own: blind A/B vs original", name, lang, None, None, f"<= {t.ab_max_original_preferred}", Status.NOT_MEASURED))
    else:
        share = fmean(a.original_preferred for a in answers)
        enough = len(answers) >= t.ab_min_sentences
        results.append(Result(
            "own: blind A/B vs original", name, lang, share, None,
            f"<= {t.ab_max_original_preferred} over >= {t.ab_min_sentences} answers",
            (Status.PASS if share <= t.ab_max_original_preferred else Status.FAIL) if enough else Status.NOT_MEASURED,
            f"{len(answers)} answers",
        ))

    c, o = _common_sentences(cand, orig)
    cand_cer, orig_cer = pooled_rate(c, "cer"), pooled_rate(o, "cer")
    if inputs.languages[lang].whisper is None:
        results.append(Result("own: Whisper CER vs original", name, lang, None, None, f"<= original + {t.cer_margin}", Status.NOT_APPLICABLE, "language not on the Whisper allow-list"))
    else:
        results.append(_judge(cand_cer, orig_cer, cand_cer is not None and orig_cer is not None and cand_cer <= orig_cer + t.cer_margin,
                              "own: Whisper CER vs original", name, lang, f"<= original + {t.cer_margin}", f"{len(c)} shared sentences"))

    _check_reference(cand, lineup.key, "own-language candidate rows")
    similarity = mean_of(cand, "speaker_similarity")
    results.append(_judge(similarity, None, similarity is not None and similarity >= t.own_speaker_similarity,
                          "own: speaker similarity to original", name, lang, f">= {t.own_speaker_similarity}"))

    cand_utmos, orig_utmos = mean_of(c, "UTMOS"), mean_of(o, "UTMOS")
    results.append(_judge(cand_utmos, orig_utmos, cand_utmos is not None and orig_utmos is not None and cand_utmos >= orig_utmos - t.own_utmos_margin,
                          "own: UTMOS vs original (recording quality only)", name, lang, f">= original - {t.own_utmos_margin}"))

    if lineup.key not in inputs.original_human:
        raise KeyError(f"{name} has no human verdict in the tiers file")
    original_passed = inputs.original_human[lineup.key] == "pass"
    verdict = next((v for v in inputs.verdicts or [] if v.key == lineup.key and v.language == lang), None)
    if not original_passed:
        results.append(Result("own: human enough (verdict page)", name, lang, None, None, "passes if the original passed", Status.PASS, "original failed the human pass"))
    elif verdict is None:
        results.append(Result("own: human enough (verdict page)", name, lang, None, None, "passes if the original passed", Status.NOT_MEASURED))
    else:
        results.append(Result("own: human enough (verdict page)", name, lang, None, None, "passes if the original passed",
                              Status.PASS if verdict.human == "pass" else Status.FAIL, f"new model: {verdict.human}"))
    return results


def cross_language(inputs: Inputs) -> list[Result]:
    t = THRESHOLDS
    own = {v.key: v.language for v in inputs.spec.lineup}
    results = []
    languages = sorted({r.espeak for r in inputs.candidate})
    for lang in languages:
        originals = _select(inputs.originals, espeak=lang)
        original_keys = sorted({r.key for r in originals})
        lineup_here = sorted({r.key for r in inputs.candidate if r.espeak == lang})

        for key in lineup_here:
            if own[key] == lang:
                continue
            cand = _select(inputs.candidate, key, lang)
            orig_lid = mean_of(originals, "lid_target")
            cand_lid = mean_of(cand, "lid_target")
            if inputs.languages[lang].voxlingua is None:
                results.append(Result("cross: language-ID confidence", str(key), lang, None, None, f">= originals - {t.lid_margin}", Status.NOT_APPLICABLE, "no VoxLingua107 label for the language"))
            else:
                results.append(_judge(cand_lid, orig_lid, cand_lid is not None and orig_lid is not None and cand_lid >= orig_lid - t.lid_margin,
                                      "cross: language-ID confidence", str(key), lang, f">= originals - {t.lid_margin}",
                                      "no original voices scored" if orig_lid is None else ""))

            if inputs.languages[lang].whisper is None:
                results.append(Result("cross: Whisper CER vs best original", str(key), lang, None, None, f"<= best original + {t.cer_margin}", Status.NOT_APPLICABLE, "language not on the Whisper allow-list"))
                continue
            best: tuple[float, VoiceKey] | None = None
            shared_n = 0
            cand_cer = None
            for okey in original_keys:
                c, o = _common_sentences(cand, _select(originals, okey))
                o_cer = pooled_rate(o, "cer")
                if o_cer is not None and (best is None or o_cer < best[0]):
                    best, cand_cer, shared_n = (o_cer, okey), pooled_rate(c, "cer"), len(c)
            if best is None:
                results.append(Result("cross: Whisper CER vs best original", str(key), lang, None, None, f"<= best original + {t.cer_margin}", Status.NOT_MEASURED, "no original voice has CER in this language"))
            else:
                results.append(_judge(cand_cer, best[0], cand_cer is not None and cand_cer <= best[0] + t.cer_margin,
                                      "cross: Whisper CER vs best original", str(key), lang, f"<= best original + {t.cer_margin}",
                                      f"best original {best[1]}, {shared_n} shared sentences"))

        passed_originals = [h for h in inputs.original_human_by_language.get(lang, []) if h == "pass"]
        verdicts = [v for v in inputs.verdicts or [] if v.language == lang]
        if not passed_originals:
            results.append(Result("cross: some lineup voice human enough", "*", lang, None, None, "one passes where an original passed", Status.PASS, "no original voice passed"))
        elif not verdicts:
            results.append(Result("cross: some lineup voice human enough", "*", lang, None, None, "one passes where an original passed", Status.NOT_MEASURED))
        else:
            passing = [str(v.key) for v in verdicts if v.human == "pass"]
            results.append(Result("cross: some lineup voice human enough", "*", lang, len(passing), None, "one passes where an original passed",
                                  Status.PASS if passing else Status.FAIL, " ".join(passing)))

        best_lineup = max((m for k in lineup_here if (m := mean_of(_select(inputs.candidate, k, lang), "UTMOS")) is not None), default=None)
        best_original = max((m for k in original_keys if (m := mean_of(_select(originals, k), "UTMOS")) is not None), default=None)
        results.append(_judge(best_lineup, best_original, best_lineup is not None and best_original is not None and best_lineup >= best_original,
                              "cross: best lineup UTMOS vs best original (within language)", "*", lang, ">= best original",
                              "no original voices scored" if best_original is None else ""))

    for lineup in inputs.spec.lineup:
        cross = [r for r in _select(inputs.candidate, lineup.key) if r.espeak != lineup.language]
        _check_reference(cross, lineup.key, "cross-language candidate rows")
        similarity = mean_of(cross, "speaker_similarity")
        results.append(_judge(similarity, None, similarity is not None and similarity >= t.cross_speaker_similarity,
                              "cross: speaker similarity to own-language output", str(lineup.key), "*", f">= {t.cross_speaker_similarity} mean"))

    for lang in sorted(inputs.spec.native_rating_languages):
        scores = [r.score for r in inputs.ratings or [] if r.language == lang]
        mean = fmean(scores) if scores else None
        results.append(_judge(mean, None, mean is not None and mean >= t.native_rating, "cross: native rating", "*", lang,
                              f">= {t.native_rating} mean; no sound consistently wrong (read the notes)", f"{len(scores)} ratings"))
    return results


def weak_language(inputs: Inputs) -> list[Result]:
    t = THRESHOLDS
    lang = inputs.spec.weak_language
    reference = _select(inputs.originals, inputs.spec.weak_reference, lang)
    if not reference:
        raise ValueError(f"no original rows for the weak-language reference {inputs.spec.weak_reference} in {lang}")
    results = []
    for lineup in inputs.spec.lineup:
        name = str(lineup.key)
        weak = _select(inputs.candidate, lineup.key, lang)
        own = _select(inputs.candidate, lineup.key, lineup.language)
        weak_utmos, own_utmos = mean_of(weak, "UTMOS"), mean_of(own, "UTMOS")
        results.append(_judge(weak_utmos, own_utmos, weak_utmos is not None and own_utmos is not None and weak_utmos >= own_utmos - t.weak_utmos_margin,
                              "weak: UTMOS vs own-language UTMOS", name, lang, f">= own language - {t.weak_utmos_margin}"))
        c, r = _common_sentences(weak, reference)
        cand_per, ref_per = pooled_rate(c, "per"), pooled_rate(r, "per")
        results.append(_judge(cand_per, ref_per, cand_per is not None and ref_per is not None and cand_per <= ref_per + t.weak_per_margin,
                              "weak: phoneme error vs reference", name, lang, f"<= {inputs.spec.weak_reference} + {t.weak_per_margin}",
                              f"{len(c)} shared sentences"))
    return results


def runtime(inputs: Inputs) -> list[Result]:
    t = THRESHOLDS
    if inputs.runtime is None:
        return [Result("runtime: CPU per audio second vs medium voice", "base", "*", None, None, f"<= {t.runtime_base}", Status.NOT_MEASURED)]
    return [
        _judge(ratio, None, ratio <= (t.runtime_base if variant == "base" else t.runtime_variant),
               "runtime: CPU per audio second vs medium voice", variant, "*", f"<= {t.runtime_base if variant == 'base' else t.runtime_variant}")
        for variant, ratio in sorted(inputs.runtime.items())
    ]


def evaluate(inputs: Inputs) -> list[Result]:
    strangers = {r.key for r in inputs.candidate} - {v.key for v in inputs.spec.lineup}
    if strangers:
        raise ValueError(f"candidate rows for voices outside the lineup: {sorted(map(str, strangers))}")
    results = [r for lineup in inputs.spec.lineup for r in own_language(inputs, lineup)]
    return results + cross_language(inputs) + weak_language(inputs) + runtime(inputs)


def needs_ears(inputs: Inputs, results: list[Result]) -> list[NeedsEars]:
    t = THRESHOLDS
    out = []
    for r in results:
        if r.status is not Status.NOT_MEASURED:
            continue
        if "human enough" in r.test:
            out.append(NeedsEars("human enough?", r.voice, r.language, "voices/verdict.html with the manifests in verdict/", r.test))
        elif "A/B" in r.test:
            out.append(NeedsEars("blind A/B", r.voice, r.language, "listening page", r.note or "no answers"))
        elif "native rating" in r.test:
            out.append(NeedsEars("native rating", r.voice, r.language, "listening page", r.note))
    for rating in inputs.ratings or []:
        if rating.note.strip():
            out.append(NeedsEars("read note: sound clearly wrong?", str(rating.key), rating.language, f"listener {rating.listener}, item {rating.item}", rating.note))
    for lineup in inputs.spec.lineup:
        out.append(NeedsEars("carried-over narrow band or crackle?", str(lineup.key), inputs.spec.weak_language, "listening (owner)", "weak-language test"))
    groups: dict[tuple[VoiceKey, str], list[ScoreRow]] = defaultdict(list)
    for r in inputs.candidate:
        groups[(r.key, r.espeak)].append(r)
    for (key, lang), rows in sorted(groups.items()):
        utmos = mean_of(rows, "UTMOS")
        if utmos is not None and utmos < t.utmos_relisten:
            out.append(NeedsEars("re-listen: low recording quality", str(key), lang, "verdict page 'clean enough'", f"UTMOS {utmos:.2f} < {t.utmos_relisten}"))
    return out


def write_verdict_manifests(out: Path, candidate: list[ScoreRow], tag: str, clips_per_voice: int = 3) -> None:
    groups: dict[tuple[VoiceKey, str], list[ScoreRow]] = defaultdict(list)
    for r in candidate:
        groups[(r.key, r.espeak)].append(r)
    by_language: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (key, lang), rows in sorted(groups.items()):
        name = verdict_voice_name(tag, key, lang)
        folder = out / lang
        folder.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(sorted(rows, key=lambda r: r.idx)[:clips_per_voice]):
            file = f"{name}__{key.speaker}__{i}.wav"
            shutil.copyfile(row.path, folder / file)
            by_language[lang].append({
                "voice": name, "speaker": key.speaker, "espeak": lang, "idx": i, "file": file,
                "sample_rate": sf.info(row.path).samplerate, "text": row.text,
            })
    for lang, rows in by_language.items():
        write_manifest(out / lang / "manifest.tsv", rows)


def _write_csv(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else f"{v:.4g}" if isinstance(v, float) else v for v in row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--originals", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--tiers", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="checkpoint name used in the verdict page's voice names")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path)
    parser.add_argument("--ab", type=Path)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if VERDICT_SEPARATOR in args.tag:
        raise SystemExit(f"--tag must not contain {VERDICT_SEPARATOR!r}")

    with args.spec.open("rb") as f:
        spec = parse_spec(tomllib.load(f))
    inputs = Inputs(
        spec=spec,
        languages=load_languages(),
        originals=[r for p in args.originals for r in read_rows(p)],
        candidate=[r for p in args.candidate for r in read_rows(p)],
        original_human=parse_original_human(args.tiers),
        original_human_by_language=original_human_by_language(args.tiers),
        verdicts=None if args.verdicts is None else parse_verdicts(args.verdicts, args.tag),
        ab=None if args.ab is None else parse_ab(args.ab),
        ratings=None if args.ratings is None else parse_ratings(args.ratings),
        runtime=None if args.runtime is None else parse_runtime(args.runtime),
    )
    results = evaluate(inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "criteria.csv", ["test", "voice", "language", "measured", "reference", "threshold", "status", "note"],
               ([r.test, r.voice, r.language, r.measured, r.reference, r.threshold, r.status.value, r.note] for r in results))
    ears = needs_ears(inputs, results)
    _write_csv(args.out / "needs_ears.csv", ["task", "voice", "language", "where", "why"],
               ([e.task, e.voice, e.language, e.where, e.why] for e in ears))
    write_verdict_manifests(args.out / "verdict", inputs.candidate, args.tag)
    counts = {s: sum(r.status is s for r in results) for s in Status}
    print(" ".join(f"{s.value}={n}" for s, n in counts.items()), f"; {len(ears)} items need ears -> {args.out}")


if __name__ == "__main__":
    main()
