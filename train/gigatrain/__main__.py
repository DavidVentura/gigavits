"""Command line (imperative shell).

    python -m gigatrain prepare         --config C --run-dir R [--previous-ids R0/ids.json]
    python -m gigatrain init            --config C --run-dir R (--base-checkpoint lessac.ckpt | --random-base) --voice LANG=voice.onnx ...
    python -m gigatrain warmup          --config C --run-dir R [--resume]          (reads R/init.pt, writes R/warmup.pt)
    python -m gigatrain render          --config C --run-dir R --weights R/warmup.pt --out DIR
    python -m gigatrain train           --config C --run-dir R [--init R/warmup.pt | --resume]
    python -m gigatrain extend          --config C --run-dir R2 --source R/backend/checkpoints/last.ckpt --like NEW=OLD [--voice NEW=v.onnx]
    python -m gigatrain train-frontend  --config C --run-dir R --source backend.ckpt --language LANG [--resume]
    python -m gigatrain train-decoder   --config C --run-dir R --source backend.ckpt --speaker KEY [--resume]
    python -m gigatrain export          --checkpoint backend.ckpt [--voice-decoder decoder.ckpt ...] --out DIR
    python -m gigatrain find-batch      --config C --run-dir R --sizes 8,16,24,32 [--steps 5]
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .config import Precision, TrainConfig, load_config, parse_dataclass, to_jsonable
from .data import ShardDataModule, default_workers, load_items, select_items, split_validation
from .ids import IdMaps, build_id_maps, require_known
from .monitor import Throughput
from .records import Vocab
from .sampling import sampling_probabilities
from .training import BackEndJob, DecoderJob, FrontEndJob, PhonemeEmbedding



def _vocab(config: TrainConfig) -> Vocab:
    return Vocab.load(Path(config.data.token_table))


def _items(config: TrainConfig):
    workers = config.data.num_workers if config.data.num_workers is not None else default_workers()
    items = load_items(config.data.shards, _vocab(config), config.audio.sample_rate, max(workers, 4))
    return select_items(items, config.data, config.audio)


def _with_cli_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    data = {k: getattr(args, k) for k in ("batch_size", "num_workers", "prefetch_factor") if getattr(args, k) is not None}
    audio = {"segment_size": args.segment_size} if args.segment_size is not None else {}
    top = {k: getattr(args, k) for k in ("max_steps",) if getattr(args, k) is not None}
    if args.precision is not None:
        top["precision"] = Precision(args.precision)
    return dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, **data),
        audio=dataclasses.replace(config.audio, **audio),
        **top,
    )


def _configure_backends(config: TrainConfig) -> None:
    L.seed_everything(config.seed, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = config.cudnn_benchmark


def _trainer(config: TrainConfig, job_dir: Path, devices: int, has_validation: bool, checkpoints: bool) -> L.Trainer:
    callbacks = [Throughput(config.log_every_n_steps)]
    if checkpoints:
        # last.ckpt every epoch (resume point), plus a kept checkpoint every keep_every_n_epochs.
        callbacks.append(
            ModelCheckpoint(
                dirpath=job_dir / "checkpoints", filename="epoch={epoch}", auto_insert_metric_name=False,
                save_last=True, save_top_k=-1, every_n_epochs=config.keep_every_n_epochs,
                # Resume points must not depend on whether validation ran in the epoch.
                save_on_train_epoch_end=True,
                # An extended checkpoint written into a fresh run dir must be overwritten, not versioned.
                enable_version_counter=False,
            )
        )
    return L.Trainer(
        default_root_dir=job_dir,
        accelerator="auto",
        devices=devices,
        # Front ends absent from a batch get no gradient.
        strategy="ddp_find_unused_parameters_true" if devices > 1 else "auto",
        precision=config.precision.value,
        max_epochs=-1,
        use_distributed_sampler=False,
        log_every_n_steps=config.log_every_n_steps,
        logger=CSVLogger(job_dir, name="logs"),
        callbacks=callbacks,
        enable_checkpointing=checkpoints,
        limit_val_batches=1.0 if has_validation else 0,
        num_sanity_val_steps=0,
        benchmark=config.cudnn_benchmark,
    )


def _datamodule(config: TrainConfig, items, ids: IdMaps) -> tuple[ShardDataModule, bool]:
    train, val = split_validation(items, config.data.val_items)
    return ShardDataModule(train, val, ids, config.data, config.audio.sample_rate, config.seed), bool(val)


def cmd_prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    items = _items(config)
    previous = IdMaps.load(args.previous_ids) if args.previous_ids else None
    ids = build_id_maps((i.record for i in items), previous)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    ids.save(args.run_dir / "ids.json")
    seconds = [i.num_samples / config.audio.sample_rate for i in items]
    languages = [i.record.language for i in items]
    p = sampling_probabilities(languages, [i.record.weight for i in items], seconds)
    for language in ids.languages:
        hours = sum(s for s, lang in zip(seconds, languages) if lang == language) / 3600
        share = sum(q for q, lang in zip(p, languages) if lang == language)
        print(f"{language:12s} {hours:8.2f} h  sampled {share:6.1%}  speakers {len(ids.speakers_of(language))}")
    print(f"{len(items)} items, {len(ids.speakers)} speakers, {len(ids.languages)} languages -> {args.run_dir / 'ids.json'}")


def _parse_voices(pairs: list[str]) -> dict[str, Path]:
    voices = {}
    for pair in pairs:
        language, sep, path = pair.partition("=")
        if not sep or language in voices:
            raise SystemExit(f"--voice expects unique LANG=path pairs, got {pair!r}")
        voices[language] = Path(path)
    return voices


def cmd_init(args: argparse.Namespace) -> None:
    from .init_from_piper import initial_state, piper_checkpoint_state, random_piper_checkpoint_state

    config = load_config(args.config)
    ids = IdMaps.load(args.run_dir / "ids.json")
    vocab = _vocab(config)
    job = BackEndJob(to_jsonable(config), ids.to_json(), vocab.size)
    voices = _parse_voices(args.voice)
    missing = set(ids.languages) - set(voices)
    if missing:
        raise SystemExit(f"no --voice for languages {sorted(missing)}; name the closest family voice explicitly")
    if args.random_base:
        base = random_piper_checkpoint_state(config.model, spec_channels=config.audio.filter_length // 2 + 1)
    else:
        base = piper_checkpoint_state(args.base_checkpoint)
    state = initial_state(job.state_dict(), base, {lang: voices[lang] for lang in ids.languages}, vocab.size)
    torch.save({"state_dict": state}, args.run_dir / "init.pt")
    print(f"wrote {args.run_dir / 'init.pt'}")


def _fit(module: L.LightningModule, dm: ShardDataModule, config: TrainConfig, job_dir: Path, devices: int, has_val: bool, resume: bool) -> None:
    trainer = _trainer(config, job_dir, devices, has_val, checkpoints=True)
    last = job_dir / "checkpoints" / "last.ckpt"
    if resume and not last.exists():
        raise SystemExit(f"--resume: {last} does not exist")
    trainer.fit(module, datamodule=dm, ckpt_path=str(last) if resume else None)


def cmd_train(args: argparse.Namespace) -> None:
    config = _with_cli_overrides(load_config(args.config), args)
    _configure_backends(config)
    ids = IdMaps.load(args.run_dir / "ids.json")
    items = _items(config)
    require_known((i.record for i in items), ids)
    module = BackEndJob(to_jsonable(config), ids.to_json(), _vocab(config).size)
    if args.init is not None and args.resume:
        raise SystemExit("--init starts a run, --resume continues one; pick one")
    if args.init is not None:
        module.load_state_dict(torch.load(args.init, map_location="cpu")["state_dict"], strict=True)
    dm, has_val = _datamodule(config, items, ids)
    _fit(module, dm, config, args.run_dir / "backend", args.devices, has_val, args.resume)


def _source_job(path: Path) -> tuple[TrainConfig, dict, int, dict]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    return parse_dataclass(TrainConfig, hparams["config"]), hparams["ids"], hparams["n_vocab"], checkpoint["state_dict"]


def _job_config(args: argparse.Namespace, source: TrainConfig) -> TrainConfig:
    """The job's own data/optimisation settings with the source run's model and audio shapes."""
    config = _with_cli_overrides(load_config(args.config), args)
    return dataclasses.replace(config, model=source.model, audio=source.audio)


def _model_state(state: dict) -> dict:
    return {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}


def cmd_train_frontend(args: argparse.Namespace) -> None:
    source, ids_json, n_vocab, state = _source_job(args.source)
    config = _job_config(args, source)
    _configure_backends(config)
    ids = IdMaps.from_json(ids_json)
    items = [i for i in _items(config) if i.record.language == args.language]
    require_known((i.record for i in items), ids)
    module = FrontEndJob(to_jsonable(config), ids_json, n_vocab, [args.language], PhonemeEmbedding.FROZEN.value)
    module.model.load_state_dict(_model_state(state))
    if args.language_row_from is not None and not args.resume:
        module.copy_language_row(args.language, args.language_row_from)
    dm, has_val = _datamodule(config, items, ids)
    _fit(module, dm, config, args.run_dir / f"frontend-{args.language}", args.devices, has_val, args.resume)


def cmd_warmup(args: argparse.Namespace) -> None:
    """All front ends and the shared phoneme embedding against the frozen initial back end, so the
    main run starts from front ends that fit the shared embedding. Writes <run-dir>/warmup.pt."""
    config = _with_cli_overrides(load_config(args.config), args)
    config = dataclasses.replace(config, max_steps=config.schedule.warmup_steps)
    _configure_backends(config)
    ids = IdMaps.load(args.run_dir / "ids.json")
    items = _items(config)
    require_known((i.record for i in items), ids)
    init = torch.load(args.run_dir / "init.pt", map_location="cpu")["state_dict"]
    module = FrontEndJob(to_jsonable(config), ids.to_json(), _vocab(config).size, list(ids.languages), PhonemeEmbedding.TRAINED.value)
    module.model.load_state_dict(_model_state(init))
    dm, has_val = _datamodule(config, items, ids)
    _fit(module, dm, config, args.run_dir / "warmup", args.devices, has_val, args.resume)
    warmed = init | {f"model.{k}": v for k, v in module.model.state_dict().items()}
    torch.save({"state_dict": warmed}, args.run_dir / "warmup.pt")
    print(f"wrote {args.run_dir / 'warmup.pt'}")


def cmd_render(args: argparse.Namespace) -> None:
    """Noise-free renders of a few held-out-first sentences per language in their own speakers'
    voices with the clean condition: the by-ear and metric sanity check after warm-up."""
    import hashlib

    import soundfile as sf

    from .records import Condition

    config = load_config(args.config)
    ids = IdMaps.load(args.run_dir / "ids.json")
    job = BackEndJob(to_jsonable(config), ids.to_json(), _vocab(config).size)
    job.model.load_state_dict(_model_state(torch.load(args.weights, map_location="cpu", weights_only=False)["state_dict"]))
    model = job.model.eval()
    items = sorted(_items(config), key=lambda i: hashlib.sha1(i.record.id.encode()).digest())
    lines = ["language\tid\tspeaker\ttext\tpath"]
    for language in ids.languages:
        (args.out / language).mkdir(parents=True, exist_ok=True)
        for item in [i for i in items if i.record.language == language][: args.per_language]:
            r = item.record
            x = torch.tensor([r.phoneme_ids])
            sid, lid, cid = (torch.tensor([v]) for v in (ids.sid(r.speaker), ids.lid(language), ids.cid(Condition.CLEAN)))
            with torch.no_grad():
                audio = model.infer(x, torch.tensor([x.size(1)]), sid, lid, cid, 0.0, 1.0, 0.0, model.dec)
            path = args.out / language / f"{r.id}.wav"
            sf.write(path, audio.squeeze().numpy(), config.audio.sample_rate)
            lines.append(f"{language}\t{r.id}\t{r.speaker}\t{r.text}\t{path}")
    (args.out / "renders.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines) - 1} renders -> {args.out}")


def _parse_pairs(pairs: list[str], flag: str) -> dict[str, str]:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or key in out:
            raise SystemExit(f"{flag} expects unique KEY=VALUE pairs, got {pair!r}")
        out[key] = value
    return out


def cmd_extend(args: argparse.Namespace) -> None:
    """Grows a back-end checkpoint to the IDs of new data and writes it as <run-dir>/backend/
    checkpoints/last.ckpt, so `train --resume` in that run dir continues from it."""
    from .extend import grown_optimizer_state, grown_state
    from .init_from_piper import read_voice_front_end

    target = args.run_dir / "backend" / "checkpoints" / "last.ckpt"
    if target.exists():
        raise SystemExit(f"{target} exists; extend into a new run dir")
    checkpoint = torch.load(str(args.source), map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    old_ids = IdMaps.from_json(hparams["ids"])
    config = _with_cli_overrides(load_config(args.config), args)
    source_config = parse_dataclass(TrainConfig, hparams["config"])
    config = dataclasses.replace(config, model=source_config.model, audio=source_config.audio)
    _configure_backends(config)
    new_ids = build_id_maps((i.record for i in _items(config)), old_ids)
    like = _parse_pairs(args.like, "--like")
    voices = {lang: read_voice_front_end(Path(p)) for lang, p in _parse_pairs(args.voice, "--voice").items()}
    old_job = BackEndJob(hparams["config"], hparams["ids"], hparams["n_vocab"])
    new_job = BackEndJob(to_jsonable(config), new_ids.to_json(), hparams["n_vocab"])
    state = grown_state(checkpoint["state_dict"], new_job.state_dict(), old_ids, new_ids, like, voices)
    new_job.load_state_dict(state, strict=True)

    def names(job: BackEndJob) -> tuple[list[str], list[str]]:
        return [f"model.{n}" for n, _ in job.model.named_parameters()], [f"disc.{n}" for n, _ in job.disc.named_parameters()]

    shapes = {n: p.shape for n, p in new_job.named_parameters()}
    optimizer_states = [
        grown_optimizer_state(old, old_names, new_names, shapes)
        for old, old_names, new_names in zip(checkpoint["optimizer_states"], names(old_job), names(new_job), strict=True)
    ]
    checkpoint |= {
        "state_dict": state,
        "optimizer_states": optimizer_states,
        "hyper_parameters": {"config": to_jsonable(config), "ids": new_ids.to_json(), "n_vocab": hparams["n_vocab"]},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    new_ids.save(args.run_dir / "ids.json")
    torch.save(checkpoint, target)
    added = {f: getattr(new_ids, f)[len(getattr(old_ids, f)):] for f in ("speakers", "languages", "conditions")}
    print(f"added {added}; wrote {target}")


def cmd_train_decoder(args: argparse.Namespace) -> None:
    source, ids_json, n_vocab, state = _source_job(args.source)
    config = _job_config(args, source)
    _configure_backends(config)
    ids = IdMaps.from_json(ids_json)
    items = [i for i in _items(config) if i.record.speaker == args.speaker]
    require_known((i.record for i in items), ids)
    module = DecoderJob(to_jsonable(config), ids_json, n_vocab, args.speaker)
    module.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")})
    module.disc.load_state_dict({k.removeprefix("disc."): v for k, v in state.items() if k.startswith("disc.")})
    module.start_from_shared_decoder()
    dm, has_val = _datamodule(config, items, ids)
    from .export import file_key

    _fit(module, dm, config, args.run_dir / f"decoder-{file_key(args.speaker)}", args.devices, has_val, args.resume)


def cmd_export(args: argparse.Namespace) -> None:
    from .export import export_graphs, verify_chain

    backend = BackEndJob.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    voice_decoders = {}
    for path in args.voice_decoder:
        job = DecoderJob.load_from_checkpoint(str(path), map_location="cpu")
        if job.ids != backend.ids:
            raise SystemExit(f"{path} was trained against a different id map")
        voice_decoders[job.speaker] = job.voice_dec
    languages = tuple(args.language) if args.language else backend.ids.languages
    exported = export_graphs(backend.model, voice_decoders, backend.ids, languages, args.out, backend.config.audio.sample_rate)
    vocab = Vocab.load(Path(backend.config.data.token_table))
    probe = [vocab.bos, vocab.pad] + [t for i in range(3, 40) for t in (i, vocab.pad)] + [vocab.eos]
    for check in verify_chain(backend.model, voice_decoders, exported, probe, sid=0, cid=0, tolerance=args.tolerance):
        print(f"{check.graph_chain}: {check.samples} samples, max diff {check.max_abs_diff:.2e}")


def cmd_find_batch(args: argparse.Namespace) -> None:
    """Runs a few adversarial-phase steps (the memory peak) per batch size and reports speed and
    peak memory; stops at the first size that runs out of memory."""
    import time

    base = _with_cli_overrides(load_config(args.config), args)
    _configure_backends(base)
    ids = IdMaps.load(args.run_dir / "ids.json")
    items = _items(base)
    require_known((i.record for i in items), ids)
    vocab = _vocab(base)
    for size in [int(s) for s in args.sizes.split(",")]:
        config = dataclasses.replace(
            base,
            data=dataclasses.replace(base.data, batch_size=size, steps_per_epoch=args.steps, val_items=0),
            schedule=dataclasses.replace(base.schedule, gan_start_step=0),
            max_steps=args.steps,
        )
        module = BackEndJob(to_jsonable(config), ids.to_json(), vocab.size)
        dm, _ = _datamodule(config, items, ids)
        trainer = _trainer(config, args.run_dir / "find-batch", 1, False, checkpoints=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            trainer.fit(module, datamodule=dm)
        except torch.OutOfMemoryError:
            print(f"batch {size}: out of memory")
            return
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
        print(f"batch {size}: {args.steps / elapsed:.2f} steps/s (incl. startup), {size * args.steps / elapsed:.1f} samples/s, peak {peak:.2f} GB")
        del module, trainer, dm


def _add_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--batch-size", type=int)
    p.add_argument("--segment-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--prefetch-factor", type=int)
    p.add_argument("--precision", choices=[p.value for p in Precision])
    p.add_argument("--max-steps", type=int)
    p.add_argument("--devices", type=int, default=1)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="gigatrain")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--previous-ids", type=Path)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("init")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    base = p.add_mutually_exclusive_group(required=True)
    base.add_argument("--base-checkpoint", type=Path)
    base.add_argument("--random-base", action="store_true", help="random lessac-structure base, for testing")
    p.add_argument("--voice", action="append", default=[], help="LANG=voice.onnx, one per language")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("train")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--init", type=Path)
    p.add_argument("--resume", action="store_true")
    _add_overrides(p)
    p.set_defaults(func=cmd_train)

    for name, key, func in (("train-frontend", "--language", cmd_train_frontend), ("train-decoder", "--speaker", cmd_train_decoder)):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--source", type=Path, required=True, help="back-end checkpoint")
        p.add_argument(key, required=True)
        p.add_argument("--resume", action="store_true")
        if name == "train-frontend":
            p.add_argument("--language-row-from", help="start the language-embedding row from this language's")
        _add_overrides(p)
        p.set_defaults(func=func)

    p = sub.add_parser("warmup")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    _add_overrides(p)
    p.set_defaults(func=cmd_warmup)

    p = sub.add_parser("render")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True, help="warmup.pt, init.pt or a back-end checkpoint")
    p.add_argument("--per-language", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("extend")
    p.add_argument("--config", type=Path, required=True, help="config whose shards hold the old and the new data")
    p.add_argument("--run-dir", type=Path, required=True, help="new run dir")
    p.add_argument("--source", type=Path, required=True, help="back-end checkpoint to grow")
    p.add_argument("--like", action="append", default=[], help="NEW_LANG=EXISTING_LANG, one per new language")
    p.add_argument("--voice", action="append", default=[], help="NEW_LANG=voice.onnx: text encoder and duration predictor from a Piper voice")
    _add_overrides(p)
    p.set_defaults(func=cmd_extend)

    p = sub.add_parser("export")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--voice-decoder", type=Path, action="append", default=[])
    p.add_argument("--language", action="append", help="front ends to export (default: all)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("find-batch")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--sizes", default="8,16,24,32,48,64")
    p.add_argument("--steps", type=int, default=5)
    _add_overrides(p)
    p.set_defaults(func=cmd_find_batch)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
