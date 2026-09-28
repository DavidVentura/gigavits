"""espeak phonemes of text through the production front end (piper-rs + espeak-ng 1.52 data), via
the `phoneme-coverage` binary built by coverage/run.sh. Its output is exactly what the shared
tokenizer (tokens/tokenizer.py) is specified against."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from config import PiperRsPhonemizer


class PhonemizeError(RuntimeError):
    pass


def phonemize(config: PiperRsPhonemizer, requests: Sequence[tuple[str, str]]) -> list[str]:
    """(espeak voice, text) -> piper-rs espeak string, in request order."""
    for espeak, text in requests:
        if "\t" in text or "\n" in text or "\t" in espeak:
            raise PhonemizeError(f"{espeak}: text holds a tab or newline: {text!r}")
    # The binary numbers sentences per espeak voice in input order.
    positions: dict[tuple[str, int], int] = {}
    per_voice: dict[str, int] = {}
    for n, (espeak, _) in enumerate(requests):
        positions[(espeak, per_voice.get(espeak, 0))] = n
        per_voice[espeak] = per_voice.get(espeak, 0) + 1
    with tempfile.TemporaryDirectory(prefix="eval-phonemize-") as tmp:
        work = Path(tmp)
        sentences = work / "sentences.tsv"
        sentences.write_text("".join(f"{e}\t{e}\t{t}\n" for e, t in requests), encoding="utf-8")
        out = work / "out.jsonl"
        result = subprocess.run(
            [str(config.binary), str(config.espeak_data), str(config.reference_config),
             str(config.reference_model), str(work / "configs"), str(sentences), str(out)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise PhonemizeError(f"{config.binary} failed ({result.returncode}): {result.stderr[-2000:]}")
        phonemes: list[str | None] = [None] * len(requests)
        for line in out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            phonemes[positions[(record["run"], record["idx"])]] = record["phonemes"]
    missing = [requests[i] for i, p in enumerate(phonemes) if p is None]
    if missing:
        raise PhonemizeError(f"no phonemes for {missing[:3]}")
    return phonemes
