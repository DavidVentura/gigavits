"""Fetch 3 individual utterances per speaker from Catalan TTS corpora, one file at a time (never whole shards).

Festcat and LaFrescat are parquet-only on the Hub, so single utterances come from the datasets-server rows API.
OpenSLR69 (projecte-aina trimmed/denoised) has per-file wavs; transcripts come from the original OpenSLR line indexes.
"""

import csv
import io
import json
import sys
import urllib.request
from dataclasses import dataclass

import numpy as np
import soundfile

from samples_common import Sample, write_sample

ROWS_API = "https://datasets-server.huggingface.co/rows?dataset=projecte-aina/{dataset}&config=default&split=train&offset={offset}&length=100"
SLR69_CLIP = "https://huggingface.co/datasets/projecte-aina/openslr-slr69-ca-trimmed-denoised/resolve/main/data/clips/{file_id}.wav"
SLR69_INDEX = "https://www.openslr.org/resources/69/line_index_{gender}.tsv"

PER_SPEAKER = 3
MIN_CHARS, MAX_CHARS = 60, 140


@dataclass(frozen=True)
class RowsSource:
    source: str
    dataset: str
    # speaker -> offset of a rows page that contains that speaker
    speaker_pages: dict[str, list[int]]


FESTCAT = RowsSource("festcat", "festcat_trimmed_denoised", {"ona": [3000], "pau": [8000], "eva": [1900, 2000], "pep": [6100]})
LAFRESCAT = RowsSource("lafrescat", "LaFrescat", {s: [0, 100, 200, 300] for s in ["olga", "grau", "emma", "lluc"]})
SLR69_SPEAKERS = ["caf_03655", "caf_09901", "cam_00459", "cam_02689"]


def http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sample_rate


def fetch_rows_source(src: RowsSource) -> None:
    pages = {
        offset: [r["row"] for r in json.loads(http_get(ROWS_API.format(dataset=src.dataset, offset=offset)))["rows"]]
        for offset in sorted({o for offsets in src.speaker_pages.values() for o in offsets})
    }
    for speaker, offsets in src.speaker_pages.items():
        rows = [row for offset in offsets for row in pages[offset]]
        picked = [r for r in rows if r["speaker_id"] == speaker and MIN_CHARS <= len(r["transcription"]) <= MAX_CHARS][:PER_SPEAKER]
        assert len(picked) == PER_SPEAKER, f"{src.source}/{speaker}: only {len(picked)} usable rows in pages {offsets}"
        for idx, row in enumerate(picked):
            audio, sample_rate = decode_wav(http_get(row["audio"][0]["src"]))
            write_sample(Sample(src.source, speaker, idx, row["transcription"].strip(), audio, sample_rate))
            print(src.source, speaker, idx, sample_rate, f"{len(audio) / sample_rate:.2f}s")


def fetch_slr69() -> None:
    transcripts = {
        file_id: text
        for gender in ["female", "male"]
        for file_id, text in csv.reader(io.StringIO(http_get(SLR69_INDEX.format(gender=gender)).decode()), delimiter="\t", quoting=csv.QUOTE_NONE)
    }
    for speaker in SLR69_SPEAKERS:
        candidates = sorted(f for f, t in transcripts.items() if f.startswith(speaker) and MIN_CHARS <= len(t) <= MAX_CHARS)
        for idx, file_id in enumerate(candidates[:PER_SPEAKER]):
            audio, sample_rate = decode_wav(http_get(SLR69_CLIP.format(file_id=file_id)))
            write_sample(Sample("slr69", speaker.replace("_", ""), idx, transcripts[file_id].strip(), audio, sample_rate))
            print("slr69", speaker, idx, sample_rate, f"{len(audio) / sample_rate:.2f}s")


def main() -> None:
    for step in sys.argv[1:] or ["festcat", "lafrescat", "slr69"]:
        {"festcat": lambda: fetch_rows_source(FESTCAT), "lafrescat": lambda: fetch_rows_source(LAFRESCAT), "slr69": fetch_slr69}[step]()


if __name__ == "__main__":
    main()
