"""Full path on local teachers: plan -> render -> filter -> shards -> verify (a few seconds of audio)."""
import dataclasses
import json
from pathlib import Path

import pytest
import soundfile

from gigagen import catalog
from gigagen.__main__ import build_jobs, generate, package
from gigagen.config import load
from gigagen.plan import selected
from gigagen.verify import parse_record, verify

LAPTOP = Path(__file__).resolve().parents[1] / "configs" / "laptop.toml"
VOICES = frozenset({"en_GB-alan-medium", "is_IS-bui-medium", "es_ES-carlfm-x_low", "mms-az"})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    config = load(LAPTOP)
    if not config.paths.bucket.exists() or not config.paths.teacher_rt.exists():
        pytest.skip("needs the local app bucket and a built teacher-rt")
    out = tmp_path_factory.mktemp("gen")
    config = dataclasses.replace(
        config,
        paths=dataclasses.replace(config.paths, out=out),
        selection=dataclasses.replace(config.selection, voices=VOICES),
        hours=dataclasses.replace(config.hours, cap_seconds=25.0),
        workers=4,
    )
    speakers = catalog.load(config.paths)
    chosen, _ = selected(speakers, config.selection)
    jobs = build_jobs(config, speakers, chosen)
    generate(config, jobs)
    package(config, jobs)
    return config


def _records(config):
    for shard in sorted((config.paths.out / "shards").glob("shard-*")):
        for line in (shard / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            yield shard, parse_record(line)


def test_verify_passes(run):
    verify(run)


def test_teacher_kinds(run):
    by_teacher = {}
    for shard, r in _records(run):
        by_teacher.setdefault(r.teacher, []).append(r)
    assert set(by_teacher) == VOICES
    assert all(r.durations is not None for r in by_teacher["en_GB-alan-medium"])
    assert all(r.durations is None for r in by_teacher["mms-az"])
    # 16 kHz teacher: durations rescaled to 22.05 kHz frames, tagged narrow band
    assert all(r.durations is not None and r.condition.value == "narrow_band" for r in by_teacher["es_ES-carlfm-x_low"])
    # espeak writes Icelandic voiceless sonorants as '#', which the old voice map drops
    table = json.loads((Path(__file__).resolve().parents[2] / "tokens" / "table.json").read_text())
    voiceless = next(t["id"] for t in table["tokens"] if t["key"] == "̊")
    lossy = [r for r in by_teacher["is_IS-bui-medium"] if voiceless in r.phoneme_ids]
    assert lossy and all(r.durations is None for r in lossy)


def test_flac_is_22050_mono(run):
    for shard, r in _records(run):
        info = soundfile.info(shard / r.audio)
        assert (info.samplerate, info.channels) == (22050, 1)
