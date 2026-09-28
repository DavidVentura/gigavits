"""Expose durations for every voice in a voice list and verify them against the unmodified model.

Usage: verify_all.py <voice_list.tsv> <results.csv> [workers]
"""

import csv
import json
import os
import sys
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from expose_durations import expose_durations

HOP = 256
LENGTH_SCALES = (0.5, 1.0, 1.7)
SEQUENCE_PHONEMES = (1, 12, 60, 220)
SPECIALS = ("^", "$", "_")
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_W = 0.8
TIMING_RUNS = 5


@dataclass(frozen=True)
class Voice:
    name: str
    onnx_path: Path
    config_path: Path


@dataclass
class Result:
    voice: str
    sample_rate: int
    quality: str
    phoneme_type: str
    num_speakers: int
    status: str
    detail: str = ""
    ceil_node: str = ""
    durations_shape: str = ""
    cases: int = 0
    checker_original: str = ""
    checker_modified: str = ""
    audio_identical: bool = False
    sum_matches_hop: bool = False
    noisy_sum_matches_hop: bool = False
    noise_is_stochastic: bool = False
    frac_pad_zero: float = float("nan")
    bos_frames_mean: float = float("nan")
    eos_frames_mean: float = float("nan")
    orig_ms: float = float("nan")
    modified_ms: float = float("nan")


def load_voices(path: str) -> list[Voice]:
    voices = []
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        name, config = line.split("\t")
        voices.append(Voice(name, Path(config.removesuffix(".json")), Path(config)))
    return voices


def id_sequences(config: dict, rng: np.random.Generator) -> list[np.ndarray]:
    id_map = config["phoneme_id_map"]
    pad, bos, eos = (id_map[s][0] for s in ("_", "^", "$"))
    symbols = np.array(sorted({ids[0] for sym, ids in id_map.items() if sym not in SPECIALS}), dtype=np.int64)
    sequences = []
    for n in SEQUENCE_PHONEMES:
        body = np.stack([rng.choice(symbols, n), np.full(n, pad)], axis=1).reshape(-1)
        sequences.append(np.concatenate([[bos, pad], body, [eos]]).astype(np.int64))
    return sequences


def feeds(ids: np.ndarray, noise_scale: float, length_scale: float, noise_w: float, sid: int | None) -> dict:
    feed = {
        "input": ids[None, :],
        "input_lengths": np.array([ids.size], dtype=np.int64),
        "scales": np.array([noise_scale, length_scale, noise_w], dtype=np.float32),
    }
    if sid is not None:
        feed["sid"] = np.array([sid], dtype=np.int64)
    return feed


def session(model: str | bytes) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])


def checker_verdict(model: onnx.ModelProto) -> str:
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        return str(e).splitlines()[0]
    return "ok"


def median_ms(sess: ort.InferenceSession, feed: dict) -> float:
    sess.run(None, feed)
    times = []
    for _ in range(TIMING_RUNS):
        start = time.perf_counter()
        sess.run(None, feed)
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000)


def verify(voice: Voice) -> Result:
    config = json.loads(voice.config_path.read_text())
    result = Result(
        voice=voice.name,
        sample_rate=config["audio"]["sample_rate"],
        quality=config["audio"].get("quality", ""),
        phoneme_type=config.get("phoneme_type", ""),
        num_speakers=config.get("num_speakers", 1),
        status="",
    )
    if not voice.onnx_path.exists():
        result.status = "no_onnx"
        result.detail = f"{voice.onnx_path} missing"
        return result

    model = onnx.load(voice.onnx_path)
    result.checker_original = checker_verdict(model)
    try:
        modified = expose_durations(model)
    except RuntimeError as e:
        result.status = "not_found"
        result.detail = str(e)
        return result
    result.checker_modified = checker_verdict(modified)
    result.ceil_node = next(n.input[0] for n in modified.graph.node if n.name == "durations/Squeeze")

    fd, tmp_path = tempfile.mkstemp(suffix=".onnx", prefix=f"{voice.name}-", dir="/tmp")
    os.close(fd)
    try:
        onnx.save(modified, tmp_path)
        original = session(str(voice.onnx_path))
        exposed = session(tmp_path)
    finally:
        os.remove(tmp_path)

    has_sid = any(i.name == "sid" for i in original.get_inputs())
    speakers = [0, result.num_speakers - 1] if has_sid else [None]
    pad = config["phoneme_id_map"]["_"][0]
    rng = np.random.default_rng(zlib.crc32(voice.name.encode()))
    sequences = id_sequences(config, rng)

    failures = []
    shapes = set()
    pad_zero, pad_total, bos_frames, eos_frames = 0, 0, [], []
    for sid in speakers:
        for ids in sequences:
            for length_scale in LENGTH_SCALES:
                feed = feeds(ids, 0.0, length_scale, 0.0, sid)
                (audio_ref,) = original.run(None, feed)
                audio, durations = exposed.run(None, feed)
                result.cases += 1
                shapes.add(durations.shape)
                case = f"sid={sid} n={ids.size} ls={length_scale}"
                if durations.shape != (1, ids.size):
                    failures.append(f"{case}: durations shape {durations.shape}")
                if not np.array_equal(audio, audio_ref):
                    failures.append(f"{case}: audio differs (max abs {np.abs(audio - audio_ref).max()})")
                if durations.min() < 0:
                    failures.append(f"{case}: negative duration")
                if int(durations.sum()) * HOP != audio.shape[-1]:
                    failures.append(f"{case}: sum={int(durations.sum())} * {HOP} != {audio.shape[-1]} samples")
                pad_positions = ids == pad
                pad_zero += int((durations[0][pad_positions] == 0).sum())
                pad_total += int(pad_positions.sum())
                bos_frames.append(int(durations[0, 0]))
                eos_frames.append(int(durations[0, -1]))

    noisy_feed = feeds(sequences[2], DEFAULT_NOISE_SCALE, 1.0, DEFAULT_NOISE_W, speakers[0])
    noisy_a, noisy_d = exposed.run(None, noisy_feed)
    noisy_b, _ = exposed.run(None, noisy_feed)
    result.noise_is_stochastic = noisy_a.shape != noisy_b.shape or not np.array_equal(noisy_a, noisy_b)
    result.noisy_sum_matches_hop = int(noisy_d.sum()) * HOP == noisy_a.shape[-1]
    if not result.noisy_sum_matches_hop:
        failures.append(f"noisy: sum={int(noisy_d.sum())} * {HOP} != {noisy_a.shape[-1]} samples")

    timing_feed = feeds(sequences[2], 0.0, 1.0, 0.0, speakers[0])
    result.orig_ms = median_ms(original, timing_feed)
    result.modified_ms = median_ms(exposed, timing_feed)

    result.durations_shape = ";".join(str(s) for s in sorted(shapes))
    result.audio_identical = not any("audio differs" in f for f in failures)
    result.sum_matches_hop = not any("samples" in f and not f.startswith("noisy") for f in failures)
    result.frac_pad_zero = pad_zero / pad_total
    result.bos_frames_mean = float(np.mean(bos_frames))
    result.eos_frames_mean = float(np.mean(eos_frames))
    if result.checker_modified != result.checker_original:
        failures.append(f"checker regressed: {result.checker_modified}")
    result.status = "fail" if failures else "pass"
    result.detail = " | ".join(failures[:5])
    return result


def main() -> None:
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    voices = load_voices(sys.argv[1])
    workers = int(sys.argv[3]) if len(sys.argv) == 4 else 4
    with ProcessPoolExecutor(workers) as pool, open(sys.argv[2], "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(Result.__dataclass_fields__))
        writer.writeheader()
        for result in pool.map(verify, voices):
            row = asdict(result)
            writer.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
            out.flush()
            print(result.voice, result.status, result.detail, flush=True)


if __name__ == "__main__":
    main()
