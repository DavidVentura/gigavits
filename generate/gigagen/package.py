"""Packaging (PLAN 2.5): FLAC items from finished jobs -> ~1 GB shards with manifest.jsonl."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .plan import unit_hash
from .render import JobStats, job_dir


class PackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class Item:
    record: dict
    source: Path
    size: int


@dataclass(frozen=True)
class TeacherReport:
    teacher: str
    rendered: int
    kept: int
    kept_hours: float
    dropped: dict[str, int]
    unaligned: int
    flagged: bool


def assign_shards(items: list[Item], shard_bytes: int, seed: int) -> list[list[Item]]:
    """Items in a fixed pseudo-random order, so each shard mixes languages and speakers, cut into
    consecutive shards of at most shard_bytes (a single larger item gets a shard of its own)."""
    ordered = sorted(items, key=lambda i: unit_hash(seed, i.record["id"]))
    shards: list[list[Item]] = []
    size = 0
    for item in ordered:
        if not shards or size + item.size > shard_bytes:
            shards.append([])
            size = 0
        shards[-1].append(item)
        size += item.size
    return shards


def teacher_reports(stats: list[JobStats], flag_fraction: float) -> list[TeacherReport]:
    by_teacher: dict[str, list[JobStats]] = defaultdict(list)
    for s in stats:
        by_teacher[s.teacher].append(s)
    out = []
    for teacher, ss in sorted(by_teacher.items()):
        rendered = sum(s.rendered for s in ss)
        kept = sum(s.kept for s in ss)
        dropped: dict[str, int] = defaultdict(int)
        for s in ss:
            for reason, n in s.dropped.items():
                dropped[reason] += n
        out.append(
            TeacherReport(
                teacher=teacher,
                rendered=rendered,
                kept=kept,
                kept_hours=sum(s.kept_seconds for s in ss) / 3600,
                dropped=dict(dropped),
                unaligned=sum(s.unaligned for s in ss),
                flagged=rendered > 0 and (rendered - kept) / rendered > flag_fraction,
            )
        )
    return out


def read_items(out: Path, job_ids: list[str]) -> list[Item]:
    items = []
    for job_id in job_ids:
        d = job_dir(out, job_id)
        if not (d / "done.json").exists():
            raise PackageError(f"job {job_id} has not finished")
        for line in (d / "items.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            source = d / record["audio"]
            items.append(Item(record, source, source.stat().st_size))
    return items


def write_shards(out: Path, shards: list[list[Item]]) -> Path:
    root = out / "shards"
    if root.exists():
        raise PackageError(f"{root} exists; move it away to repackage")
    for n, shard in enumerate(shards):
        d = root / f"shard-{n:05d}"
        (d / "audio").mkdir(parents=True)
        with (d / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
            for item in shard:
                rel = f"audio/{item.record['id']}.flac"
                # hard links: the job directories stay intact and nothing is copied
                os.link(item.source, d / rel)
                manifest.write(json.dumps({**item.record, "audio": rel}, ensure_ascii=False) + "\n")
    return root


def write_reports(out: Path, reports: list[TeacherReport], stats: list[JobStats]) -> None:
    d = out / "report"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "filter.tsv").open("w", encoding="utf-8") as f:
        f.write("teacher\trendered\tkept\tkept_hours\tduration_outlier\tinternal_silence\tunaligned\tflagged\n")
        for r in reports:
            f.write(
                f"{r.teacher}\t{r.rendered}\t{r.kept}\t{r.kept_hours:.3f}\t{r.dropped.get('duration_outlier', 0)}\t"
                f"{r.dropped.get('internal_silence', 0)}\t{r.unaligned}\t{r.flagged}\n"
            )
    by_speaker: dict[tuple[str, str], list[JobStats]] = defaultdict(list)
    for st in stats:
        by_speaker[(st.language, st.speaker)].append(st)
    with (d / "speakers.tsv").open("w", encoding="utf-8") as f:
        f.write("language\tspeaker\tkept\tkept_hours\n")
        for (language, speaker), ss in sorted(by_speaker.items()):
            f.write(f"{language}\t{speaker}\t{sum(x.kept for x in ss)}\t{sum(x.kept_seconds for x in ss) / 3600:.3f}\n")
