"""Client for the Rust teacher runtime (generate/rt): app-identical phonemization and MNN teachers."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class RuntimeFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class PiperFeed:
    fed: str  # the trimmed NFD string piper-rs maps to IDs (espeak, or the text for `text` voices)
    piper_ids: tuple[int, ...]  # without BOS/PAD/EOS


@dataclass(frozen=True)
class Phonemized:
    phonemes: str  # untrimmed espeak string, as piper-rs `espeak_phonemize` returns it
    teacher: PiperFeed | None


@dataclass(frozen=True)
class Rendered:
    phonemes: str
    audio: np.ndarray  # float32, as piper-rs returns it (peak-normalized)
    sample_rate: int


class TeacherRuntime:
    def __init__(self, binary: Path, espeak_data: Path, spec: dict, log: Path):
        read_fd, write_fd = os.pipe()
        self._log = log.open("ab")
        self._proc = subprocess.Popen(
            [str(binary), str(espeak_data), json.dumps(spec), str(write_fd)],
            stdin=subprocess.PIPE,
            stdout=self._log,
            stderr=self._log,
            pass_fds=(write_fd,),
        )
        os.close(write_fd)
        self._responses = os.fdopen(read_fd, "rb")
        self._spec = spec

    def _call(self, request: dict) -> dict:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
            self._proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeFailed(f"teacher-rt died ({self._spec}); see {self._log.name}") from e
        line = self._responses.readline()
        if not line:
            self._proc.wait()
            raise RuntimeFailed(f"teacher-rt exited {self._proc.returncode} ({self._spec}); see {self._log.name}")
        return json.loads(line)

    def phonemize(self, text: str) -> Phonemized:
        r = self._call({"text": text, "render": None})
        t = r["teacher"]
        feed = None if t is None else PiperFeed(fed=t["fed"], piper_ids=tuple(t["piper_ids"]))
        return Phonemized(phonemes=r["phonemes"], teacher=feed)

    def render(self, text: str, speed: float, scratch: Path) -> Rendered:
        r = self._call({"text": text, "render": {"speed": speed, "audio_out": str(scratch)}})
        audio = np.fromfile(scratch, dtype="<f4")
        scratch.unlink()
        if audio.size != r["audio"]["samples"]:
            raise RuntimeFailed(f"expected {r['audio']['samples']} samples, read {audio.size}")
        return Rendered(phonemes=r["phonemes"], audio=audio, sample_rate=r["audio"]["sample_rate"])

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        code = self._proc.wait()
        self._responses.close()
        self._log.close()
        if code != 0:
            raise RuntimeFailed(f"teacher-rt exited {code} ({self._spec})")

    def __enter__(self) -> TeacherRuntime:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        self._proc.kill()
        self._proc.wait()
        self._responses.close()
        self._log.close()
