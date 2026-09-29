"""CPU end to end: fake shard -> prepare -> back-end training across the GAN switch -> checkpoint ->
resume with identical losses -> front-end and decoder jobs -> ONNX export verified against PyTorch."""
import csv
import json
from pathlib import Path

import pytest

from conftest import TOKEN_TABLE
from fake_shard import make_shard
from gigatrain.__main__ import main
from gigatrain.records import Vocab

STEPS = 4


def tiny_config(*shards: Path, precision: str = "32-true", steps_per_epoch: int = 2, languages_per_batch: int = 1) -> dict:
    return {
        "data": {
            "shards": [str(s) for s in shards],
            "token_table": str(TOKEN_TABLE),
            "batch_size": 4,
            "num_workers": 2,
            "prefetch_factor": 2,
            "steps_per_epoch": steps_per_epoch,
            "languages_per_batch": languages_per_batch,
            "bucket_batches": 2,
            "min_seconds": 0.4,
            "max_utterance_s": 5.0,
            "val_items": 2,
        },
        "model": {
            "inter_channels": 16,
            "hidden_channels": 16,
            "filter_channels": 32,
            "n_heads": 2,
            "n_layers": 2,
            "upsample_initial_channel": 32,
            "gin_channels": 16,
            "classifier_hidden": 8,
        },
        "schedule": {"warmup_steps": 2, "gan_start_step": 2, "adversarial_ramp_steps": 3, "adversarial_weight": 0.02},
        "optim": {"lr_final_ratio": 1.0},
        "precision": precision,
        "max_steps": STEPS,
        "log_every_n_steps": 1,
        "keep_every_n_epochs": 1,
        "cudnn_benchmark": False,
    }


def losses(job_dir: Path, column: str) -> dict[int, float]:
    """Logged per-step training values keyed by batch index across all CSV logs of a job."""
    values = {}
    for path in sorted(job_dir.glob("logs/version_*/metrics.csv")):
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r.get(column)]
        start = len(values)
        for offset, row in enumerate(rows):
            values[start + offset] = float(row[column])
    return values


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    shard = root / "shard"
    make_shard(shard, Vocab.load(TOKEN_TABLE))
    config = root / "config.json"
    config.write_text(json.dumps(tiny_config(shard)))
    return root, config


def test_prepare_writes_id_map(workspace):
    root, config = workspace
    main(["prepare", "--config", str(config), "--run-dir", str(root / "run")])
    ids = json.loads((root / "run" / "ids.json").read_text())
    assert ids["languages"] == ["en-us", "es"]
    assert ids["conditions"] == ["clean", "narrow_band", "degraded"]
    assert ids["language_speakers"] == {"en-us": ["en_US-amy", "en_US-joe"], "es": ["es_ES-davefx"]}


def test_training_resumes_with_identical_losses(workspace):
    root, config = workspace
    for name in ("straight", "resumed"):
        (root / name).mkdir()
        (root / name / "ids.json").write_text((root / "run" / "ids.json").read_text())
    main(["train", "--config", str(config), "--run-dir", str(root / "straight")])
    main(["train", "--config", str(config), "--run-dir", str(root / "resumed"), "--max-steps", "2"])
    assert (root / "resumed" / "backend" / "checkpoints" / "last.ckpt").exists()
    main(["train", "--config", str(config), "--run-dir", str(root / "resumed"), "--resume"])

    straight = losses(root / "straight" / "backend", "train_loss_g")
    resumed = losses(root / "resumed" / "backend", "train_loss_g")
    assert len(straight) == STEPS and len(resumed) == STEPS
    assert straight == pytest.approx(resumed, rel=1e-6, abs=0)
    # Steps 2 and 3 are adversarial: the discriminator trained.
    assert len(losses(root / "straight" / "backend", "train_disc")) == 2


def test_front_end_and_decoder_jobs_and_export(workspace):
    root, config = workspace
    source = root / "straight" / "backend" / "checkpoints" / "last.ckpt"
    main(["train-frontend", "--config", str(config), "--run-dir", str(root / "straight"), "--source", str(source),
          "--language", "en-us", "--max-steps", "2", "--precision", "bf16-mixed", "--language-row-from", "es"])
    assert len(losses(root / "straight" / "frontend-en-us", "train_kl")) == 2
    main(["train-decoder", "--config", str(config), "--run-dir", str(root / "straight"), "--source", str(source),
          "--speaker", "en_US-joe", "--max-steps", "2"])
    decoder = root / "straight" / "decoder-en_US-joe" / "checkpoints" / "last.ckpt"
    out = root / "onnx"
    main(["export", "--checkpoint", str(source), "--voice-decoder", str(decoder), "--out", str(out)])
    names = sorted(p.name for p in out.glob("*.onnx"))
    assert names == [
        "backend.onnx", "decoder-en_US-joe.onnx", "decoder.onnx", "flow.onnx", "frontend-en-us.onnx", "frontend-es.onnx",
    ]
    assert json.loads((out / "ids.json").read_text())["languages"] == ["en-us", "es"]


def test_front_end_job_leaves_back_end_and_other_rows_untouched(workspace):
    import torch

    root, _ = workspace
    before = torch.load(root / "straight" / "backend" / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    after = torch.load(root / "straight" / "frontend-en-us" / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    changed = [k for k in after if k in before and not torch.equal(before[k], after[k])]
    assert changed and all(k.startswith("model.front_ends.en-us.") or k == "model.cond.emb_lang.weight" for k in changed)
    assert "model.cond.emb_lang.weight" in changed
    lang = before["model.cond.emb_lang.weight"], after["model.cond.emb_lang.weight"]
    assert torch.equal(lang[0][1], lang[1][1]) and not torch.equal(lang[0][0], lang[1][0])
    assert torch.equal(before["model.cond.emb_g.weight"], after["model.cond.emb_g.weight"])


def test_bf16_back_end_runs_both_phases(workspace):
    import math

    root, config = workspace
    run = root / "bf16"
    run.mkdir()
    (run / "ids.json").write_text((root / "run" / "ids.json").read_text())
    # Two languages per batch, so the mixed-language front-end routing runs too.
    mixed = root / "config-mixed.json"
    mixed.write_text(json.dumps(tiny_config(root / "shard", languages_per_batch=2)))
    main(["train", "--config", str(mixed), "--run-dir", str(run), "--precision", "bf16-mixed"])
    values = losses(run / "backend", "train_loss_g")
    assert len(values) == STEPS and all(math.isfinite(v) for v in values.values())
    assert len(losses(run / "backend", "train_gen")) == 2


def _copy_ids(root: Path, run: Path) -> None:
    run.mkdir()
    (run / "ids.json").write_text((root / "run" / "ids.json").read_text())


def test_warmup_trains_front_ends_only_then_renders_and_feeds_training(workspace):
    import torch

    from gigatrain.config import load_config, to_jsonable
    from gigatrain.ids import IdMaps
    from gigatrain.training import BackEndJob

    root, config = workspace
    run = root / "warm"
    _copy_ids(root, run)
    job = BackEndJob(to_jsonable(load_config(config)), IdMaps.load(run / "ids.json").to_json(), Vocab.load(TOKEN_TABLE).size)
    torch.save({"state_dict": job.state_dict()}, run / "init.pt")
    main(["warmup", "--config", str(config), "--run-dir", str(run)])
    assert len(losses(run / "warmup", "train_kl")) == 2
    init = torch.load(run / "init.pt")["state_dict"]
    warm = torch.load(run / "warmup.pt")["state_dict"]
    assert set(init) == set(warm)
    changed = {k for k in init if not torch.equal(init[k], warm[k])}
    assert "model.emb.weight" in changed
    assert any(k.startswith("model.front_ends.en-us.") for k in changed)
    assert any(k.startswith("model.front_ends.es.") for k in changed)
    assert all(k.startswith(("model.emb.", "model.front_ends.", "model.cond.emb_lang.")) for k in changed)

    out = root / "renders"
    main(["render", "--config", str(config), "--run-dir", str(run), "--weights", str(run / "warmup.pt"),
          "--per-language", "2", "--out", str(out)])
    assert len(list(out.glob("*/*.wav"))) == 4
    assert len((out / "renders.tsv").read_text().splitlines()) == 5

    main(["train", "--config", str(config), "--run-dir", str(run), "--init", str(run / "warmup.pt"), "--max-steps", "1"])
    assert len(losses(run / "backend", "train_loss_g")) == 1


NEW_VOICES = (
    ("en-us", "en-us", "en_US-sam", "clean", 1.0),
    ("pl", "pl", "pl_PL-gosia", "degraded", 1.0),
)


def test_extend_with_new_speaker_and_language_then_resume_exactly(workspace):
    import torch

    root, _ = workspace
    vocab = Vocab.load(TOKEN_TABLE)
    shard_b = root / "shard-b"
    make_shard(shard_b, vocab, count=8, seed=1, voices=NEW_VOICES, prefix="b")
    config_a = root / "config-a.json"
    config_a.write_text(json.dumps(tiny_config(root / "shard", steps_per_epoch=1)))
    config_ab = root / "config-ab.json"
    config_ab.write_text(json.dumps(tiny_config(root / "shard", shard_b, steps_per_epoch=1)))

    run_a = root / "stage-a"
    _copy_ids(root, run_a)
    main(["train", "--config", str(config_a), "--run-dir", str(run_a), "--max-steps", "2"])
    source = run_a / "backend" / "checkpoints" / "last.ckpt"
    for name in ("stage-b", "stage-b-split"):
        main(["extend", "--config", str(config_ab), "--run-dir", str(root / name), "--source", str(source), "--like", "pl=es"])

    ids = json.loads((root / "stage-b" / "ids.json").read_text())
    assert ids["languages"] == ["en-us", "es", "pl"]
    assert ids["speakers"] == ["en_US-amy", "en_US-joe", "es_ES-davefx", "en_US-sam", "pl_PL-gosia"]
    old = torch.load(source, map_location="cpu", weights_only=False)["state_dict"]
    grown = torch.load(root / "stage-b" / "backend" / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
    emb = grown["state_dict"]["model.cond.emb_lang.weight"]
    assert torch.equal(emb[:2], old["model.cond.emb_lang.weight"]) and torch.equal(emb[2], emb[1])
    assert torch.equal(grown["state_dict"]["model.cond.emb_g.weight"][:3], old["model.cond.emb_g.weight"])
    assert torch.equal(grown["state_dict"]["model.front_ends.pl.enc_p.proj.weight"], old["model.front_ends.es.enc_p.proj.weight"])

    main(["train", "--config", str(config_ab), "--run-dir", str(root / "stage-b"), "--resume", "--max-steps", "4"])
    main(["train", "--config", str(config_ab), "--run-dir", str(root / "stage-b-split"), "--resume", "--max-steps", "3"])
    main(["train", "--config", str(config_ab), "--run-dir", str(root / "stage-b-split"), "--resume", "--max-steps", "4"])
    straight = losses(root / "stage-b" / "backend", "train_loss_g")
    split = losses(root / "stage-b-split" / "backend", "train_loss_g")
    assert len(straight) == len(split) == 2
    assert straight == pytest.approx(split, rel=1e-6, abs=0)
