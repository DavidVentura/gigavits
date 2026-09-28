"""Teacher-data generator entry point.

    python -m gigagen <config.toml> plan|select|fetch-check|fetch|generate|package|verify|all

plan: speakers, hours and tempo; select: sentence lists for every language; fetch-check: HEAD every
model URL the run needs; fetch: download them into the bucket mirror; generate: render,
filter and write FLAC per job (resumable); package: shards + reports; verify: re-read the shards.
`all` runs generate, package and verify on the sentence lists already selected.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from pathlib import Path

from . import catalog, fetch
from .catalog import (
    CoquiSource, GlowTtsSource, KokoroJaSource, KokoroSource, Mimic3Source, MmsSource, PiperSource, Speaker,
)
from .config import Config, load
from .package import assign_shards, read_items, teacher_reports, write_reports, write_shards
from .plan import Job, hours_for, jobs_for, selected, tempo_factors
from .render import (
    JobStats, RenderContext, convert_model, engine_conversions, job_dir, patch_kokoro, prepare_model, run_job,
)
from .sentences import load_lists, run_selection
from .verify import verify

# Real-time factor of one job (Piper: onnxruntime fp32, one thread; others: piper-rs + MNN, from
# bench/); only used to start the slowest jobs first.
REALTIME = {
    PiperSource: 13.0, KokoroSource: 4.2, KokoroJaSource: 4.2, MmsSource: 4.7,
    CoquiSource: 25.0, Mimic3Source: 23.0, GlowTtsSource: 3.2,
}

COMMANDS = ("plan", "select", "fetch-check", "fetch", "generate", "package", "verify", "all")


def build_jobs(config: Config, speakers: list[Speaker], chosen: list[Speaker]) -> list[Job]:
    tempo = tempo_factors(speakers, config.render.rate)
    jobs = []
    lists = {e: load_lists(config.paths.sentences, e) for e in {s.espeak for s in chosen}}
    for s in chosen:
        jobs += jobs_for(
            s, lists[s.espeak].train, config.hours, tempo, config.render.part_hours,
            config.sentences.core, config.render.second_render_fraction, config.render.seed,
        )
    return jobs


def print_plan(config: Config, speakers: list[Speaker], chosen: list[Speaker], corpus: list[Speaker]) -> None:
    tempo = tempo_factors(speakers, config.render.rate)
    by_language: dict[str, float] = defaultdict(float)
    for s in chosen:
        h = hours_for(s, config.hours)
        if config.hours.cap_seconds is not None:
            h = min(h, config.hours.cap_seconds / 3600)
        by_language[s.language] += h
        print(f"{s.language}\t{s.key}\t{s.tier.value}\t{s.condition.value}\t{type(s.source).__name__}\t"
              f"hours={h:.3f}\ttempo={tempo.get(s.key, float('nan')):.3f}")
    print(f"# {len(chosen)} speakers, {sum(by_language.values()):.2f} h over {len(by_language)} languages")
    print("# " + " ".join(f"{k}={v:.2f}" for k, v in sorted(by_language.items())))
    if corpus:
        print(f"# {len(corpus)} recorded-speech speakers are left to the corpus loaders: "
              + " ".join(s.key for s in corpus))


def generate(config: Config, jobs: list[Job]) -> None:
    ctx = RenderContext(paths=config.paths, noise=config.render.noise, quality=config.quality, threads=config.threads_per_worker)
    piper = {j.speaker.voice: j.speaker.source for j in jobs if isinstance(j.speaker.source, PiperSource)}
    conversions = {
        mnn: onnx
        for j in jobs if not isinstance(j.speaker.source, PiperSource)
        for onnx, mnn in engine_conversions(j.speaker, config.paths).values()
    }
    if any(isinstance(j.speaker.source, (KokoroSource, KokoroJaSource)) for j in jobs):
        patch_kokoro(config.paths.kokoro_source_onnx, config.paths.kokoro_onnx)
    with ProcessPoolExecutor(config.workers) as pool:
        list(pool.map(prepare_model, piper.values(), [config.paths.out] * len(piper), piper.keys()))
        list(pool.map(convert_model, [config.paths.teacher_rt] * len(conversions), conversions.values(), conversions.keys()))
        ordered = sorted(jobs, key=lambda j: -j.budget_seconds / REALTIME[type(j.speaker.source)])
        pending = {pool.submit(run_job, j, ctx) for j in ordered}
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_EXCEPTION)
                for f in done:
                    stats = f.result()
                    print(f"done {stats.job_id}: {stats.kept}/{stats.rendered} kept, {stats.kept_seconds:.1f} s", flush=True)
        except BaseException:
            # finished jobs keep their done.json; a rerun resumes from them
            pool.shutdown(cancel_futures=True)
            raise


def package(config: Config, jobs: list[Job]) -> None:
    job_ids = [j.job_id for j in jobs]
    stats = [
        JobStats(**json.loads((job_dir(config.paths.out, i) / "done.json").read_text(encoding="utf-8")))
        for i in job_ids
    ]
    items = read_items(config.paths.out, job_ids)
    root = write_shards(config.paths.out, assign_shards(items, config.shard_bytes, config.render.seed))
    reports = teacher_reports(stats, config.quality.flag_drop_fraction)
    write_reports(config.paths.out, reports, stats)
    for r in reports:
        if r.flagged:
            print(f"FLAGGED {r.teacher}: dropped {r.rendered - r.kept}/{r.rendered} {r.dropped}")
    print(f"{len(items)} items in {root}")


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[1] not in COMMANDS:
        sys.exit(__doc__)
    config = load(Path(argv[0]))
    command = argv[1]
    # the mirror must exist before the catalog can read the voices' configs
    if command == "fetch-check":
        fetch.check(config)
        return
    if command == "fetch":
        fetch.fetch(config)
        return
    speakers = catalog.load(config.paths)
    chosen, corpus = selected(speakers, config.selection)
    if command == "plan":
        print_plan(config, speakers, chosen, corpus)
        return
    if command == "select":
        # every language at once, whatever the run renders: held-out sentences are shared across
        # espeak voices reading the same text, so lists must not depend on the speaker selection
        run_selection(config, frozenset(s.espeak for s in speakers if s.espeak not in config.selection.exclude_espeak))
        return
    if command == "verify":
        verify(config)
        return
    jobs = build_jobs(config, speakers, chosen)
    if command in ("generate", "all"):
        generate(config, jobs)
    if command in ("package", "all"):
        package(config, jobs)
    if command == "all":
        verify(config)


if __name__ == "__main__":
    main(sys.argv[1:])
