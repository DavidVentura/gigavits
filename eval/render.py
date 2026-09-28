"""Pluggable renderers: anything that turns (language, espeak phonemes, seed) into audio for one
voice. The baseline renders the original Piper voices through PiperOnnxRenderer; the new model
gets its own Renderer implementation and goes through the same baseline/scoring path."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np
import onnxruntime as ort

from manifest import VoiceKey
from repo import tokens_module


@dataclass(frozen=True)
class Scales:
    noise: float
    length: float
    noise_w: float


PIPER_DEFAULTS = Scales(noise=0.667, length=1.0, noise_w=0.8)


@dataclass(frozen=True)
class RenderRequest:
    espeak: str
    phonemes: str


@dataclass(frozen=True)
class Rendered:
    audio: np.ndarray
    sample_rate: int
    cpu_s: float


class Renderer(Protocol):
    """One voice with one noise seed; renders in sequence are reproducible for the same seed and
    the same request order."""

    key: VoiceKey
    espeak: str

    def render(self, request: RenderRequest) -> Rendered: ...


@dataclass
class CpuTime:
    seconds: float = 0.0


@contextmanager
def cpu_timer() -> Iterator[CpuTime]:
    """Process CPU time of the block: with a single-threaded session this is the runtime hook's
    CPU seconds, comparable with bench/ and PLAN's per-audio-second figures."""
    spent = CpuTime()
    start = time.process_time()
    try:
        yield spent
    finally:
        spent.seconds = time.process_time() - start


class PiperOnnxRenderer:
    def __init__(self, onnx_path: Path, key: VoiceKey, seed: int, scales: Scales = PIPER_DEFAULTS):
        config = json.loads(onnx_path.with_name(onnx_path.name + ".json").read_text(encoding="utf-8"))
        self.key = key
        self.espeak = config["espeak"]["voice"]
        self.sample_rate = int(config["audio"]["sample_rate"])
        self.phoneme_id_map = config["phoneme_id_map"]
        self.scales = scales
        speakers = config.get("speaker_id_map") or {}
        if config["num_speakers"] > 1:
            if key.speaker not in speakers:
                raise KeyError(f"{onnx_path.name}: no speaker {key.speaker!r} (has {sorted(speakers)[:10]}...)")
            self.sid: int | None = speakers[key.speaker]
        else:
            if key.speaker != "0":
                raise KeyError(f"{onnx_path.name} is single-speaker; speaker must be '0', got {key.speaker!r}")
            self.sid = None
        # The graph's RandomNormalLike kernels draw their seed when the session is created and then
        # advance per run, so the seed has to be set here rather than per render.
        ort.set_seed(seed)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])

    def render(self, request: RenderRequest) -> Rendered:
        if request.espeak != self.espeak:
            raise ValueError(f"{self.key} speaks {self.espeak}, asked for {request.espeak}")
        teacher = tokens_module("remap").teacher_input(request.phonemes, self.phoneme_id_map)
        inputs = {
            "input": np.array([teacher.ids], dtype=np.int64),
            "input_lengths": np.array([len(teacher.ids)], dtype=np.int64),
            "scales": np.array([self.scales.noise, self.scales.length, self.scales.noise_w], dtype=np.float32),
        }
        if self.sid is not None:
            inputs["sid"] = np.array([self.sid], dtype=np.int64)
        with cpu_timer() as spent:
            audio = self.session.run(None, inputs)[0].squeeze().astype(np.float32)
        return Rendered(audio, self.sample_rate, spent.seconds)


def find_piper_onnx(bucket: Path, voice: str) -> Path:
    matches = sorted(bucket.rglob(f"{voice}.onnx"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{voice}.onnx: {len(matches)} matches under {bucket}")
    return matches[0]
