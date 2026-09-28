"""Evaluation models, each imported and loaded on first use, so a config that leaves a metric out
never imports or downloads its model. All take 16 kHz mono float32 audio."""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np

from config import Config, EcapaConfig, Hub, Local, ModelSource, RandomInit, WhisperConfig
from repo import QUALITY, quality_score

SAMPLE_RATE = 16000


def _name(source: Hub | Local) -> str:
    return source.name if isinstance(source, Hub) else str(source.path)


class WhisperTranscriber:
    def __init__(self, config: WhisperConfig):
        from faster_whisper import WhisperModel

        self.config = config
        self.model = WhisperModel(_name(config.model), device=config.device, compute_type=config.compute_type)

    def __call__(self, audio: np.ndarray, language: str) -> str:
        segments, _ = self.model.transcribe(
            audio, language=language, beam_size=self.config.beam_size, condition_on_previous_text=False
        )
        return " ".join(s.text.strip() for s in segments)


def ctc_collapse(ids: list[int], blank: int) -> list[int]:
    out = []
    previous = None
    for i in ids:
        if i != previous and i != blank:
            out.append(i)
        previous = i
    return out


class PhonemeRecognizer:
    """wav2vec2 phoneme CTC, greedy decoding straight from vocab.json (the model's tokenizer class
    would pull in the `phonemizer` package just to decode)."""

    SPECIAL = ("<pad>", "<s>", "</s>", "<unk>")

    def __init__(self, source: ModelSource, device: str):
        import torch
        from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        if isinstance(source, RandomInit):
            vocab = json.loads((source.vocab_dir / "vocab.json").read_text(encoding="utf-8"))
            torch.manual_seed(0)
            self.model = Wav2Vec2ForCTC(Wav2Vec2Config(
                vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                intermediate_size=64, conv_dim=(32, 32), conv_stride=(5, 64), conv_kernel=(10, 64),
                num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, pad_token_id=vocab["<pad>"],
            ))
            self.features = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0, do_normalize=True)
        else:
            name = _name(source)
            self.model = Wav2Vec2ForCTC.from_pretrained(name)
            self.features = Wav2Vec2FeatureExtractor.from_pretrained(name)
            vocab = json.loads(Path(self._vocab_file(source)).read_text(encoding="utf-8"))
        self.device = device
        self.model.to(device).eval()
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.blank = vocab["<pad>"]
        self.special = {vocab[t] for t in self.SPECIAL if t in vocab}

    @staticmethod
    def _vocab_file(source: Hub | Local) -> str:
        if isinstance(source, Local):
            return str(source.path / "vocab.json")
        from huggingface_hub import hf_hub_download

        return hf_hub_download(source.name, "vocab.json")

    def __call__(self, audio: np.ndarray) -> list[str]:
        import torch

        inputs = self.features(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits[0].cpu()
        ids = ctc_collapse(logits.argmax(dim=-1).tolist(), self.blank)
        return [self.id_to_token[i] for i in ids if i not in self.special]


def _ecapa(config: EcapaConfig):
    from speechbrain.inference.classifiers import EncoderClassifier

    return EncoderClassifier.from_hparams(source=_name(config.model), savedir=str(config.savedir), run_opts={"device": config.device})


class SpeakerEncoder:
    def __init__(self, config: EcapaConfig):
        self.classifier = _ecapa(config)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            embedding = self.classifier.encode_batch(torch.from_numpy(audio)[None, :])[0, 0].cpu().numpy()
        return embedding / np.linalg.norm(embedding)


class LanguageIdentifier:
    def __init__(self, config: EcapaConfig):
        self.classifier = _ecapa(config)
        encoder = self.classifier.hparams.label_encoder
        # VoxLingua107 labels read "en: English".
        self.codes = [encoder.ind2lab[i].split(":")[0].strip() for i in range(len(encoder.ind2lab))]

    def __call__(self, audio: np.ndarray) -> dict[str, float]:
        import torch

        with torch.inference_mode():
            log_posteriors = self.classifier.classify_batch(torch.from_numpy(audio)[None, :])[0][0]
        posteriors = log_posteriors.exp().cpu().numpy()
        if not np.isclose(posteriors.sum(), 1.0, atol=1e-3):
            raise RuntimeError(f"language-ID output is not a log posterior (sums to {posteriors.sum()})")
        return dict(zip(self.codes, posteriors.tolist()))


class RecordingQuality:
    """UTMOS22-strong and DNSMOS exactly as calibrated in quality/score.py."""

    def __init__(self):
        module = quality_score()
        models = QUALITY / "models"
        self.dnsmos = module.Dnsmos(models)
        self.utmos = module.Utmos(models)

    def __call__(self, audio: np.ndarray) -> dict[str, float]:
        return self.dnsmos(audio) | self.utmos(audio)


class Models:
    def __init__(self, config: Config):
        self.config = config

    @cached_property
    def transcriber(self) -> WhisperTranscriber:
        assert self.config.cer is not None
        return WhisperTranscriber(self.config.cer)

    @cached_property
    def phonemes(self) -> PhonemeRecognizer:
        assert self.config.per is not None
        return PhonemeRecognizer(self.config.per.model, self.config.per.device)

    @cached_property
    def speaker(self) -> SpeakerEncoder:
        assert self.config.speaker is not None
        return SpeakerEncoder(self.config.speaker.encoder)

    @cached_property
    def lid(self) -> LanguageIdentifier:
        assert self.config.lid is not None
        return LanguageIdentifier(self.config.lid)

    @cached_property
    def quality(self) -> RecordingQuality:
        assert self.config.quality is not None
        return RecordingQuality()
