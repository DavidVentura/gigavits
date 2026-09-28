import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile

OUT_DIR = Path("/home/david/git/gigapiper/voices/candidates/ca")
SENTENCES_TSV = Path("/home/david/git/gigapiper/coverage/work/sentences.tsv")
SAMPLE_RATE = 22050
MANIFEST_HEADER = ["voice", "speaker", "espeak", "idx", "file", "sample_rate", "phonemes", "text"]


@dataclass(frozen=True)
class Sample:
    source: str
    speaker: str
    idx: int
    text: str
    audio: np.ndarray
    sample_rate: int

    @property
    def voice(self) -> str:
        return f"{self.source}-{self.speaker}".replace(" ", "")

    @property
    def file_name(self) -> str:
        return f"{self.voice}__{self.speaker}__{self.idx}.wav"


def load_sentences() -> list[str]:
    """The 3 Catalan sentences starting at the 25th length percentile (same as voices/samples)."""
    with SENTENCES_TSV.open() as f:
        texts = [row[2] for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if row[0] == "ca"]
    by_length = sorted(texts, key=len)
    start = int(len(by_length) * 0.25)
    return by_length[start : start + 3]


def write_sample(sample: Sample) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(sample.audio, -1.0, 1.0) * 32767).astype(np.int16)
    soundfile.write(OUT_DIR / sample.file_name, pcm, sample.sample_rate, subtype="PCM_16")

    manifest = OUT_DIR / "manifest.tsv"
    rows = []
    if manifest.exists():
        with manifest.open() as f:
            rows = [r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if r["file"] != sample.file_name]
    rows.append(
        {
            "voice": sample.voice,
            "speaker": "0",
            "espeak": "ca",
            "idx": str(sample.idx),
            "file": sample.file_name,
            "sample_rate": str(sample.sample_rate),
            "phonemes": str(sum(c.isalpha() for c in sample.text)),
            "text": sample.text,
        }
    )
    rows.sort(key=lambda r: (r["voice"], int(r["idx"])))
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
