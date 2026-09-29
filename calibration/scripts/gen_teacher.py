"""Synthesize teacher audio with Piper voices on CPU and write a multi-speaker
piper1-gpl dataset (dataset_type=phoneme_ids: wav|speaker|text|ids).

Measures CPU generation speed (audio seconds per wall second).
"""

import argparse
import csv
import itertools
import json
import time
import wave
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import onnxruntime
from piper import PiperVoice, SynthesisConfig

NUM_SYMBOLS = 256


@dataclass(frozen=True)
class Speaker:
    name: str
    voice: str
    voice_speaker_id: int | None
    run: str
    first_sentence: int


@dataclass(frozen=True)
class Task:
    speaker: Speaker
    index: int
    text: str


@dataclass(frozen=True)
class Result:
    wav_name: str
    speaker: str
    text: str
    phoneme_ids: list[int]
    audio_seconds: float
    synth_seconds: float


SPEAKERS = [
    Speaker("alan", "en_GB-alan-medium", None, "en-gb-x-rp", 0),
    Speaker("alba", "en_GB-alba-medium", None, "en-gb-x-rp", 200),
    Speaker("sharvard_0", "es_ES-sharvard-medium", 0, "es", 0),
    Speaker("sharvard_1", "es_ES-sharvard-medium", 1, "es", 200),
    Speaker("thorsten", "de_DE-thorsten-medium", None, "de", 0),
]

_VOICES: dict[str, PiperVoice] = {}
_ARGS: argparse.Namespace


def init_worker(args: argparse.Namespace) -> None:
    global _ARGS
    _ARGS = args


def load_voice(name: str) -> PiperVoice:
    if name in _VOICES:
        return _VOICES[name]
    model_path = _ARGS.voices_dir / f"{name}.onnx"
    voice = PiperVoice.load(model_path)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = _ARGS.threads
    options.inter_op_num_threads = 1
    voice.session = onnxruntime.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    _VOICES[name] = voice
    return voice


def synthesize(task: Task) -> Result:
    voice = load_voice(task.speaker.voice)
    config = SynthesisConfig(speaker_id=task.speaker.voice_speaker_id)
    t0 = time.perf_counter()
    chunks = list(voice.synthesize(task.text, syn_config=config))
    synth_seconds = time.perf_counter() - t0
    wav_name = f"{task.speaker.name}_{task.index:04d}.wav"
    with wave.open(str(_ARGS.out_dir / "wavs" / wav_name), "wb") as wav:
        wav.setframerate(chunks[0].sample_rate)
        wav.setsampwidth(chunks[0].sample_width)
        wav.setnchannels(chunks[0].sample_channels)
        for chunk in chunks:
            wav.writeframes(chunk.audio_int16_bytes)
    num_samples = sum(len(chunk.audio_float_array) for chunk in chunks)
    phoneme_ids = list(itertools.chain.from_iterable(chunk.phoneme_ids for chunk in chunks))
    if max(phoneme_ids) >= NUM_SYMBOLS:
        raise ValueError(f"phoneme id {max(phoneme_ids)} >= {NUM_SYMBOLS} in {wav_name}")
    return Result(
        wav_name=wav_name,
        speaker=task.speaker.name,
        text=task.text,
        phoneme_ids=phoneme_ids,
        audio_seconds=num_samples / chunks[0].sample_rate,
        synth_seconds=synth_seconds,
    )


def load_sentences(path: Path) -> dict[str, list[str]]:
    by_run: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for run, _app_lang, text in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            by_run.setdefault(run, []).append(text)
    return by_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--voices-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-speaker", type=int, required=True)
    parser.add_argument("--speakers", nargs="*", default=[s.name for s in SPEAKERS])
    parser.add_argument("--procs", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True, help="onnxruntime intra-op threads per process")
    args = parser.parse_args()

    by_run = load_sentences(args.sentences)
    speakers = [s for s in SPEAKERS if s.name in args.speakers]
    tasks = [
        Task(speaker, i, by_run[speaker.run][speaker.first_sentence + i])
        for speaker in speakers
        for i in range(args.per_speaker)
    ]
    (args.out_dir / "wavs").mkdir(parents=True, exist_ok=True)

    # Load voices before timing so model load is not counted as generation time.
    t_load = time.perf_counter()
    with Pool(args.procs, initializer=init_worker, initargs=(args,)) as pool:
        warm = [Task(s, -1, "Warm up.") for s in speakers for _ in range(args.procs)]
        pool.map(synthesize, warm, chunksize=1)
        load_seconds = time.perf_counter() - t_load
        t0 = time.perf_counter()
        results = list(pool.imap_unordered(synthesize, tasks, chunksize=1))
        wall = time.perf_counter() - t0
    for p in (args.out_dir / "wavs").glob("*_-001.wav"):
        p.unlink()

    results.sort(key=lambda r: r.wav_name)
    with open(args.out_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for r in results:
            writer.writerow([r.wav_name, r.speaker, r.text, " ".join(map(str, r.phoneme_ids))])

    audio = sum(r.audio_seconds for r in results)
    summary = {
        "procs": args.procs,
        "threads_per_proc": args.threads,
        "utterances": len(results),
        "audio_seconds": round(audio, 1),
        "audio_hours": round(audio / 3600, 3),
        "wall_seconds": round(wall, 1),
        "warmup_and_load_seconds": round(load_seconds, 1),
        "audio_sec_per_wall_sec": round(audio / wall, 2),
        "mean_utt_seconds": round(audio / len(results), 2),
        "per_speaker_audio_seconds": {
            s.name: round(sum(r.audio_seconds for r in results if r.speaker == s.name), 1) for s in speakers
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
