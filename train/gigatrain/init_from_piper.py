"""Initial weights from a Piper training checkpoint (lessac structure) and per-language Piper voices.

Every parameter of the new model gets exactly one explicit source; an unknown parameter, a missing
source tensor, a shape mismatch or an unused checkpoint tensor raises.

- Flow, decoder, posterior encoder and discriminators: from the checkpoint.
- Conditioning layers the single-speaker checkpoint lacks (flow/posterior `cond_layer`, decoder
  `cond`, duration predictor `cond`): zero, so step 0 behaves like the checkpoint whatever g is.
  Weight-normed layers get a zero gain and keep a random direction (a zero direction would divide
  by zero).
- Shared phoneme embedding: the checkpoint's first n_vocab rows (the shared table keeps Piper's IDs).
- Each front end's text encoder and the duration-predictor parts used at inference: from that
  language's voice ONNX. The SDP's posterior flows and its second flow (`dp.post_*`, `dp.flows.1`)
  run only in training, are absent from every inference graph, and come from the checkpoint.
- Speaker/language/condition embeddings and the adversarial classifiers: fresh.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from .config import ModelShape
from .vits.models import MultiPeriodDiscriminator, SynthesizerTrn


class InitError(ValueError):
    pass


@dataclass(frozen=True)
class FromCheckpoint:
    key: str
    rows: int | None = None


@dataclass(frozen=True)
class FromVoice:
    language: str
    name: str


class Fresh(Enum):
    ZERO = "zero"
    KEEP = "keep"


Source = FromCheckpoint | FromVoice | Fresh

# SDP modules that the inference graph runs (reverse pass: flows 7, 5, 3 and the affine flow 0).
_VOICE_DP_PREFIXES = ("dp.pre.", "dp.proj.", "dp.convs.", "dp.flows.0.", "dp.flows.3.", "dp.flows.5.", "dp.flows.7.")
# The checkpoint's text encoder layers and inference SDP parts are replaced per language from voices.
_CHECKPOINT_UNUSED = re.compile(
    r"model_g\.(enc_p\.(?!emb\.weight$)|" + "|".join(re.escape(p) for p in _VOICE_DP_PREFIXES) + ").+"
)
# Multi-speaker voices condition their SDP on their own speaker table; ours starts from zero.
_VOICE_UNUSED = re.compile(r"dp\.cond\..+")
_COND = re.compile(r"(enc_q\.enc\.cond_layer|flow\.flows\.\d+\.enc\.cond_layer|dec\.cond)\.(weight|weight_g|weight_v|bias)")


def _conditioning_source(suffix: str) -> Fresh:
    return Fresh.KEEP if suffix == "weight_v" else Fresh.ZERO


def source_for(key: str, n_vocab: int) -> Source:
    if key == "model.emb.weight":
        return FromCheckpoint("model_g.enc_p.emb.weight", rows=n_vocab)
    front_end = re.fullmatch(r"model\.front_ends\.([^.]+)\.(.+)", key)
    if front_end:
        language, name = front_end.groups()
        if name.startswith("dp.cond."):
            return Fresh.ZERO
        if name.startswith("enc_p.") or name.startswith(_VOICE_DP_PREFIXES):
            return FromVoice(language, name)
        if name.startswith("dp."):
            return FromCheckpoint(f"model_g.{name}")
        raise InitError(f"no initialisation rule for {key}")
    if key.startswith(("model.cond.", "model.adversary.")):
        return Fresh.KEEP
    if key.startswith(("model.enc_q.", "model.flow.", "model.dec.")):
        name = key.removeprefix("model.")
        cond = _COND.fullmatch(name)
        if cond:
            return _conditioning_source(cond.group(2))
        return FromCheckpoint(f"model_g.{name}")
    if key.startswith("disc."):
        return FromCheckpoint(f"model_d.{key.removeprefix('disc.')}")
    raise InitError(f"no initialisation rule for {key}")


def init_plan(target_keys: list[str], n_vocab: int) -> dict[str, Source]:
    return {key: source_for(key, n_vocab) for key in target_keys}


def _resolve(key: str, source: Source, current: torch.Tensor, checkpoint: dict, voices: dict) -> torch.Tensor:
    if source is Fresh.KEEP:
        return current
    if source is Fresh.ZERO:
        return torch.zeros_like(current)
    if isinstance(source, FromCheckpoint):
        if source.key not in checkpoint:
            raise InitError(f"{key}: checkpoint has no {source.key}")
        value = checkpoint[source.key]
        if source.rows is not None:
            if value.shape[0] < source.rows:
                raise InitError(f"{key}: {source.key} has {value.shape[0]} rows, need {source.rows}")
            value = value[: source.rows]
        return value
    if source.language not in voices:
        raise InitError(f"{key}: no voice given for language {source.language!r}")
    weights = voices[source.language]
    if source.name not in weights:
        raise InitError(f"{key}: voice for {source.language!r} has no {source.name}")
    return torch.from_numpy(np.array(weights[source.name]))


def apply_plan(
    plan: dict[str, Source],
    target: dict[str, torch.Tensor],
    checkpoint: dict[str, torch.Tensor],
    voices: dict[str, dict[str, np.ndarray]],
) -> dict[str, torch.Tensor]:
    if set(plan) != set(target):
        raise InitError("plan does not cover exactly the target parameters")
    out = {}
    for key, source in plan.items():
        value = _resolve(key, source, target[key], checkpoint, voices)
        if tuple(value.shape) != tuple(target[key].shape):
            raise InitError(f"{key}: source {source} has shape {tuple(value.shape)}, model has {tuple(target[key].shape)}")
        out[key] = value.to(target[key].dtype).clone()
    used_checkpoint = {s.key for s in plan.values() if isinstance(s, FromCheckpoint)}
    unused = sorted(k for k in checkpoint if k not in used_checkpoint and not _CHECKPOINT_UNUSED.fullmatch(k))
    if unused:
        raise InitError(f"checkpoint tensors without a destination: {unused[:10]}")
    for language, weights in voices.items():
        used = {s.name for s in plan.values() if isinstance(s, FromVoice) and s.language == language}
        unused = sorted(n for n in weights if n not in used and not _VOICE_UNUSED.fullmatch(n))
        if unused:
            raise InitError(f"voice for {language!r} has tensors without a destination: {unused[:10]}")
    return out


def _constant(name: str, graph: onnx.GraphProto, initializers: dict[str, np.ndarray]) -> np.ndarray:
    """Value of a tensor computed from constants only (initializer, Constant, Exp, Neg)."""
    if name in initializers:
        return initializers[name]
    producers = [n for n in graph.node if name in n.output]
    if len(producers) != 1:
        raise InitError(f"{name} has {len(producers)} producers")
    node = producers[0]
    if node.op_type == "Constant":
        (attribute,) = node.attribute
        if attribute.name != "value":
            raise InitError(f"{name}: Constant with {attribute.name}")
        return numpy_helper.to_array(attribute.t)
    if node.op_type == "Exp":
        return np.exp(_constant(node.input[0], graph, initializers))
    if node.op_type == "Neg":
        return -_constant(node.input[0], graph, initializers)
    raise InitError(f"{name} is computed by {node.op_type}, not from constants")


def _elementwise_affine_logs(graph: onnx.GraphProto, initializers: dict[str, np.ndarray]) -> np.ndarray:
    # ElementwiseAffine's reverse pass is (x - m) * exp(-logs). Exporters fold exp(-logs) into an
    # anonymous constant, fully or partly, so logs is recovered from the Mul after the Sub of m.
    subs = [n for n in graph.node if n.op_type == "Sub" and list(n.input)[1:] == ["dp.flows.0.m"]]
    if len(subs) != 1:
        raise InitError(f"expected one Sub of dp.flows.0.m, found {len(subs)}")
    muls = [n for n in graph.node if n.op_type == "Mul" and subs[0].output[0] in n.input]
    if len(muls) != 1:
        raise InitError(f"expected one Mul after dp.flows.0.m, found {len(muls)}")
    (other,) = [i for i in muls[0].input if i != subs[0].output[0]]
    scale = _constant(other, graph, initializers)
    if (scale <= 0).any():
        raise InitError("exp(-logs) of dp.flows.0 is not positive")
    return -np.log(scale)


def read_voice_front_end(path: Path) -> dict[str, np.ndarray]:
    """Text-encoder and inference SDP weights of a Piper voice ONNX, keyed by upstream names."""
    graph = onnx.load(str(path)).graph
    initializers = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    weights = {
        name: value
        for name, value in initializers.items()
        if name.startswith(("enc_p.", "dp.")) and not name.startswith("enc_p.emb.")
    }
    if "dp.flows.0.logs" not in weights:
        weights["dp.flows.0.logs"] = _elementwise_affine_logs(graph, initializers)
    return weights


def piper_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    return dict(checkpoint["state_dict"])


def random_piper_checkpoint_state(shape: ModelShape, n_vocab: int = 256, spec_channels: int = 513, seed: int = 0) -> dict[str, torch.Tensor]:
    """A randomly initialised single-speaker Piper training state dict with lessac's structure, for
    exercising the init path without downloading the real checkpoint."""
    torch.manual_seed(seed)
    generator = SynthesizerTrn(
        n_vocab=n_vocab,
        spec_channels=spec_channels,
        segment_size=32,
        inter_channels=shape.inter_channels,
        hidden_channels=shape.hidden_channels,
        filter_channels=shape.filter_channels,
        n_heads=shape.n_heads,
        n_layers=shape.n_layers,
        kernel_size=shape.kernel_size,
        p_dropout=shape.p_dropout,
        resblock=shape.resblock,
        resblock_kernel_sizes=shape.resblock_kernel_sizes,
        resblock_dilation_sizes=shape.resblock_dilation_sizes,
        upsample_rates=shape.upsample_rates,
        upsample_initial_channel=shape.upsample_initial_channel,
        upsample_kernel_sizes=shape.upsample_kernel_sizes,
        n_speakers=1,
        gin_channels=0,
        use_sdp=True,
    )
    state = {f"model_g.{k}": v for k, v in generator.state_dict().items()}
    state |= {f"model_d.{k}": v for k, v in MultiPeriodDiscriminator().state_dict().items()}
    return state


def initial_state(
    target: dict[str, torch.Tensor],
    checkpoint: dict[str, torch.Tensor],
    voice_paths: dict[str, Path],
    n_vocab: int,
) -> dict[str, torch.Tensor]:
    voices = {language: read_voice_front_end(path) for language, path in voice_paths.items()}
    return apply_plan(init_plan(list(target), n_vocab), target, checkpoint, voices)
