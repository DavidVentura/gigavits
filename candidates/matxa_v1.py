"""Render Matxa-TTS v1 multiaccent (projecte-aina, e2e WaveNext ONNX) samples for the Catalan candidate listening test.

Text front-end replicates projecte-aina/matxa-alvocat-tts-ca (text/cleaners.py, text/symbols.py):
lowercase -> phonemizer espeak (accent voice, stress, punctuation) -> symbol ids interspersed with 0.
Run with ESPEAK_DATA_PATH pointing at an espeak-ng-data that has ca-ba/ca-nw/ca-va.
"""

import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime
from phonemizer.backend import EspeakBackend

from samples_common import SAMPLE_RATE, Sample, load_sentences, write_sample

MODEL = Path(__file__).parent / "models" / "matxa_multiaccent_wavenext_e2e.onnx"

_PAD = "_"
_PUNCTUATION = ';:,.!?¡¿—…"«»“” '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
SYMBOL_TO_ID = {s: i for i, s in enumerate([_PAD, *_PUNCTUATION, *_LETTERS, *_LETTERS_IPA])}

# (name, speaker id in the multiaccent model, espeak voice used by the matching cleaner)
SPEAKERS = [
    ("olga", 1, "ca-ba"),
    ("grau", 2, "ca"),
    ("emma", 5, "ca-nw"),
    ("lluc", 6, "ca-va"),
]
TEMPERATURE = 0.2
LENGTH_SCALE = 0.89


def phoneme_ids(backend: EspeakBackend, text: str) -> np.ndarray:
    phonemes = re.sub(r"\s+", " ", backend.phonemize([text.lower()], strip=True)[0])
    ids = [SYMBOL_TO_ID[c] for c in phonemes if c in SYMBOL_TO_ID]
    interspersed = [0] * (len(ids) * 2 + 1)
    interspersed[1::2] = ids
    return np.array([interspersed], dtype=np.int64)


def main() -> None:
    session = onnxruntime.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    sentences = load_sentences()
    for name, speaker_id, espeak_voice in SPEAKERS:
        backend = EspeakBackend(espeak_voice, preserve_punctuation=True, with_stress=True)
        for idx, text in enumerate(sentences):
            x = phoneme_ids(backend, text)
            _, wav = session.run(
                None,
                {
                    "x": x,
                    "x_lengths": np.array([x.shape[1]], dtype=np.int64),
                    "scales": np.array([TEMPERATURE, LENGTH_SCALE], dtype=np.float32),
                    "spks": np.array([speaker_id], dtype=np.int64),
                },
            )
            write_sample(Sample(source="matxa1", speaker=name, idx=idx, text=text, audio=wav[0], sample_rate=SAMPLE_RATE))
            print(name, idx, f"{wav.shape[1] / SAMPLE_RATE:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
