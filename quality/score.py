"""Score TTS sample clips with DNSMOS (P.835 + P.808) and UTMOS22-strong, aggregated per speaker.

Usage:
    venv/bin/python score.py <manifest.tsv> <out_scores.csv> [--clips-out clips.csv]

The manifest's WAV `file` column is resolved relative to the manifest's directory.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
MODEL_SR = 16000
DNSMOS_WINDOW_S = 9.01
METRICS = ("SIG", "BAK", "OVRL", "P808", "UTMOS")


@dataclass(frozen=True)
class Clip:
    voice: str
    speaker: str
    espeak: str
    idx: int
    path: Path
    sample_rate: int


@dataclass(frozen=True)
class ClipScore:
    clip: Clip
    duration_s: float
    scores: dict[str, float]


def parse_manifest(manifest: Path) -> list[Clip]:
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [
        Clip(
            voice=r["voice"],
            speaker=r["speaker"],
            espeak=r["espeak"],
            idx=int(r["idx"]),
            path=manifest.parent / r["file"],
            sample_rate=int(r["sample_rate"]),
        )
        for r in rows
    ]


def load_16k(clip: Clip) -> np.ndarray:
    audio, sr = sf.read(clip.path, dtype="float32", always_2d=True)
    if sr != clip.sample_rate:
        raise ValueError(f"{clip.path}: manifest says {clip.sample_rate} Hz, file is {sr} Hz")
    mono = audio.mean(axis=1)
    if sr == MODEL_SR:
        return mono
    return librosa.resample(mono, orig_sr=sr, target_sr=MODEL_SR, res_type="soxr_hq").astype(np.float32)


class Dnsmos:
    """Port of microsoft/DNS-Challenge DNSMOS/dnsmos_local.py (non-personalized)."""

    P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

    def __init__(self, models: Path) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.primary = ort.InferenceSession(str(models / "sig_bak_ovr.onnx"), opts)
        self.p808 = ort.InferenceSession(str(models / "model_v8.onnx"), opts)

    @staticmethod
    def melspec(audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(y=audio, sr=MODEL_SR, n_fft=321, hop_length=160, n_mels=120)
        return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T

    def __call__(self, audio: np.ndarray) -> dict[str, float]:
        window = int(DNSMOS_WINDOW_S * MODEL_SR)
        # The reference implementation loops short clips up to the model window.
        tiled = audio
        while len(tiled) < window:
            tiled = np.append(tiled, audio)
        num_hops = int(np.floor(len(tiled) / MODEL_SR) - DNSMOS_WINDOW_S) + 1
        segments = [tiled[i * MODEL_SR : i * MODEL_SR + window] for i in range(num_hops)]
        segments = [s for s in segments if len(s) == window]
        sig, bak, ovr, p808 = [], [], [], []
        for seg in segments:
            raw_sig, raw_bak, raw_ovr = self.primary.run(None, {"input_1": seg.astype(np.float32)[None, :]})[0][0]
            mel = self.melspec(seg[:-160]).astype(np.float32)[None, :, :]
            p808.append(self.p808.run(None, {"input_1": mel})[0][0][0])
            sig.append(self.P_SIG(raw_sig))
            bak.append(self.P_BAK(raw_bak))
            ovr.append(self.P_OVR(raw_ovr))
        return {"SIG": float(np.mean(sig)), "BAK": float(np.mean(bak)), "OVRL": float(np.mean(ovr)), "P808": float(np.mean(p808))}


class Utmos:
    def __init__(self, models: Path) -> None:
        torch.hub.set_dir(str(models / "torch" / "hub"))
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()

    def __call__(self, audio: np.ndarray) -> dict[str, float]:
        with torch.inference_mode():
            score = self.model(torch.from_numpy(audio)[None, :], MODEL_SR)
        return {"UTMOS": float(score[0])}


def aggregate(clip_scores: list[ClipScore]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[ClipScore]] = {}
    for cs in clip_scores:
        groups.setdefault((cs.clip.voice, cs.clip.speaker), []).append(cs)
    rows = []
    for (voice, speaker), members in groups.items():
        first = members[0].clip
        row: dict[str, object] = {
            "voice": voice,
            "speaker": speaker,
            "espeak": first.espeak,
            "sample_rate": first.sample_rate,
            "n_clips": len(members),
        }
        for m in METRICS:
            values = np.array([cs.scores[m] for cs in members])
            row[m] = round(float(values.mean()), 4)
        row["UTMOS_min"] = round(float(min(cs.scores["UTMOS"] for cs in members)), 4)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path, help="per-speaker scores CSV")
    parser.add_argument("--clips-out", type=Path, help="optional per-clip scores CSV")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    clips = parse_manifest(args.manifest)
    dnsmos = Dnsmos(MODELS)
    utmos = Utmos(MODELS)

    clip_scores = []
    for i, clip in enumerate(clips):
        audio = load_16k(clip)
        clip_scores.append(ClipScore(clip, len(audio) / MODEL_SR, dnsmos(audio) | utmos(audio)))
        print(f"\r{i + 1}/{len(clips)} {clip.path.name}", end="", file=sys.stderr)
    print(file=sys.stderr)

    write_csv(args.out, aggregate(clip_scores))
    if args.clips_out:
        write_csv(
            args.clips_out,
            [
                {"voice": cs.clip.voice, "speaker": cs.clip.speaker, "espeak": cs.clip.espeak, "idx": cs.clip.idx,
                 "sample_rate": cs.clip.sample_rate, "duration_s": round(cs.duration_s, 3)}
                | {m: round(cs.scores[m], 4) for m in METRICS}
                for cs in clip_scores
            ],
        )


if __name__ == "__main__":
    main()
