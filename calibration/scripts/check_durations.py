"""Patch each teacher voice with expose_durations.py and check the durations output on CPU."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime
from piper import PiperVoice

HOP = 256
CASES = [
    ("en_GB-alan-medium", None, "The quick brown fox jumps over the lazy dog."),
    ("en_GB-alba-medium", None, "She sells sea shells by the sea shore."),
    ("es_ES-sharvard-medium", 1, "El veloz murciélago hindú comía feliz cardillo y kiwi."),
    ("de_DE-thorsten-medium", None, "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich."),
]


def main() -> None:
    voices_dir, expose_script, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sid, text in CASES:
        src = voices_dir / f"{name}.onnx"
        dst = out_dir / f"{name}.durations.onnx"
        subprocess.run([sys.executable, str(expose_script), str(src), str(dst)], check=True)
        voice = PiperVoice.load(src)
        ids = [voice.phonemes_to_ids(p) for p in voice.phonemize(text)][0]
        session = onnxruntime.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        feeds = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        }
        if sid is not None:
            feeds["sid"] = np.array([sid], dtype=np.int64)
        audio, durations = session.run(["output", "durations"], feeds)
        frames = int(durations.sum())
        samples = audio.shape[-1]
        ok = durations.shape == (1, len(ids)) and frames * HOP == samples
        print(
            f"{name}: outputs={[o.name for o in session.get_outputs()]} ids={len(ids)} "
            f"durations_shape={durations.shape} sum_frames={frames} audio_samples={samples} "
            f"frames*hop==samples={frames * HOP == samples} {'OK' if ok else 'MISMATCH'}"
        )


if __name__ == "__main__":
    main()
