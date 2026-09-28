"""ONNX export as separate graphs, and a check that they chain to the PyTorch model's audio.

Graphs (inputs -> outputs):
- frontend-<language>.onnx: input, input_lengths, scales, sid, lid, cid -> z_p, y_mask, durations
- backend.onnx: z_p, y_mask, sid, lid, cid -> output (flow + shared decoder)
- flow.onnx: z_p, y_mask, sid, lid, cid -> z
- decoder.onnx and decoder-<speaker>.onnx: z, sid, lid, cid -> output

`scales` is Piper's [noise_scale, length_scale, noise_scale_w]; `durations` matches
durations/expose_durations.py (frames per input ID).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from .ids import IdMaps
from .model import FrontEnd, MultilingualVits, copy_decoder, copy_model
from .vits.models import Generator

OPSET_VERSION = 15


class FrontEndGraph(nn.Module):
    def __init__(self, model: MultilingualVits, front_end: FrontEnd):
        super().__init__()
        self.model = model
        self.front_end = front_end

    def forward(self, input, input_lengths, scales, sid, lid, cid):
        return self.model.infer_prior(input, input_lengths, sid, lid, cid, scales[0], scales[1], scales[2], self.front_end)


class FlowGraph(nn.Module):
    def __init__(self, model: MultilingualVits):
        super().__init__()
        self.model = model

    def forward(self, z_p, y_mask, sid, lid, cid):
        return self.model.infer_latent(z_p, y_mask, sid, lid, cid)


class DecoderGraph(nn.Module):
    def __init__(self, model: MultilingualVits, decoder: Generator):
        super().__init__()
        self.model = model
        self.decoder = decoder

    def forward(self, z, sid, lid, cid):
        return self.model.infer_audio(z, sid, lid, cid, self.decoder)


class BackEndGraph(nn.Module):
    def __init__(self, model: MultilingualVits, decoder: Generator):
        super().__init__()
        self.model = model
        self.decoder = decoder

    def forward(self, z_p, y_mask, sid, lid, cid):
        z = self.model.infer_latent(z_p, y_mask, sid, lid, cid)
        return self.model.infer_audio(z, sid, lid, cid, self.decoder)


def _ids(value: int) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.long)


def _export(module: nn.Module, args: tuple, path: Path, inputs: list[str], outputs: list[str], dynamic: dict) -> None:
    torch.onnx.export(
        module,
        args,
        str(path),
        opset_version=OPSET_VERSION,
        input_names=inputs,
        output_names=outputs,
        dynamic_axes=dynamic,
        dynamo=False,
    )


def file_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


@dataclass(frozen=True)
class Exported:
    front_ends: dict[str, Path]
    backend: Path
    flow: Path
    decoders: dict[str, Path]


def export_graphs(
    model: MultilingualVits,
    voice_decoders: dict[str, Generator],
    ids: IdMaps,
    languages: tuple[str, ...],
    out_dir: Path,
    sample_rate: int,
) -> Exported:
    """Exports inference copies; weight norm is folded on the decoders only, as upstream does."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model = copy_model(model).eval()
    model.dec.remove_weight_norm()
    decoders = {name: copy_decoder(dec, model.shape).eval() for name, dec in voice_decoders.items()}
    for dec in decoders.values():
        dec.remove_weight_norm()

    sid, lid, cid = _ids(0), _ids(0), _ids(0)
    phonemes = torch.randint(0, model.emb.num_embeddings, (1, 40))
    lengths = torch.tensor([40])
    scales = torch.tensor([0.667, 1.0, 0.8])
    batch_phonemes = {0: "batch", 1: "phonemes"}
    frames = {0: "batch", 2: "frames"}
    front_ends = {}
    with torch.no_grad():
        for language in languages:
            path = out_dir / f"frontend-{file_key(language)}.onnx"
            _export(
                FrontEndGraph(model, model.front_ends[language]),
                (phonemes, lengths, scales, sid, _ids(ids.lid(language)), cid),
                path,
                ["input", "input_lengths", "scales", "sid", "lid", "cid"],
                ["z_p", "y_mask", "durations"],
                {"input": batch_phonemes, "input_lengths": {0: "batch"}, "sid": {0: "batch"}, "lid": {0: "batch"},
                 "cid": {0: "batch"}, "z_p": frames, "y_mask": frames, "durations": batch_phonemes},
            )
            front_ends[language] = path
        z_p = torch.randn(1, model.shape.inter_channels, 50)
        y_mask = torch.ones(1, 1, 50)
        latent_inputs = {"z_p": frames, "y_mask": frames, "sid": {0: "batch"}, "lid": {0: "batch"}, "cid": {0: "batch"}}
        audio = {"output": {0: "batch", 2: "samples"}}
        backend = out_dir / "backend.onnx"
        _export(BackEndGraph(model, model.dec), (z_p, y_mask, sid, lid, cid), backend,
                ["z_p", "y_mask", "sid", "lid", "cid"], ["output"], latent_inputs | audio)
        flow = out_dir / "flow.onnx"
        _export(FlowGraph(model), (z_p, y_mask, sid, lid, cid), flow,
                ["z_p", "y_mask", "sid", "lid", "cid"], ["z"], latent_inputs | {"z": frames})
        decoder_paths = {}
        decoder_inputs = {"z": frames, "sid": {0: "batch"}, "lid": {0: "batch"}, "cid": {0: "batch"}}
        for name, dec in ({"": model.dec} | decoders).items():
            path = out_dir / (f"decoder-{file_key(name)}.onnx" if name else "decoder.onnx")
            _export(DecoderGraph(model, dec), (z_p, sid, lid, cid), path, ["z", "sid", "lid", "cid"], ["output"], decoder_inputs | audio)
            decoder_paths[name] = path
    ids.save(out_dir / "ids.json")
    (out_dir / "audio.json").write_text(json.dumps({"sample_rate": sample_rate}) + "\n", encoding="utf-8")
    return Exported(front_ends, backend, flow, decoder_paths)


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@dataclass(frozen=True)
class ChainCheck:
    graph_chain: str
    samples: int
    max_abs_diff: float


def verify_chain(
    model: MultilingualVits,
    voice_decoders: dict[str, Generator],
    exported: Exported,
    phoneme_ids: list[int],
    sid: int,
    cid: int,
    tolerance: float,
) -> list[ChainCheck]:
    """With noise off (noise_scale = noise_scale_w = 0) the chained ONNX graphs must reproduce the
    PyTorch model's audio for every exported front end, with the shared and every voice decoder."""
    model = copy_model(model).eval()
    voice_decoders = {name: copy_decoder(dec, model.shape).eval() for name, dec in voice_decoders.items()}
    x = torch.tensor([phoneme_ids])
    x_lengths = torch.tensor([len(phoneme_ids)])
    scales = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    backend = _session(exported.backend)
    flow = _session(exported.flow)
    decoder_sessions = {name: _session(path) for name, path in exported.decoders.items()}
    reference_decoders = {"": model.dec} | voice_decoders
    checks = []
    for language, front_path in exported.front_ends.items():
        lid = model.languages.index(language)
        feeds = {"sid": np.array([sid]), "lid": np.array([lid]), "cid": np.array([cid])}
        z_p, y_mask, _ = _session(front_path).run(
            None, {"input": x.numpy(), "input_lengths": x_lengths.numpy(), "scales": scales} | feeds
        )
        z = flow.run(None, {"z_p": z_p, "y_mask": y_mask} | feeds)[0]
        chains = [("backend", "", backend.run(None, {"z_p": z_p, "y_mask": y_mask} | feeds)[0])]
        chains += [
            (f"flow -> {exported.decoders[name].stem}", name, session.run(None, {"z": z} | feeds)[0])
            for name, session in decoder_sessions.items()
        ]
        for chain, decoder_name, actual in chains:
            with torch.no_grad():
                expected = model.infer(
                    x, x_lengths, torch.tensor([sid]), torch.tensor([lid]), torch.tensor([cid]),
                    0.0, 1.0, 0.0, reference_decoders[decoder_name],
                ).numpy()
            label = f"frontend-{language} -> {chain}"
            if actual.shape != expected.shape:
                raise AssertionError(f"{label}: shape {actual.shape} != {expected.shape}")
            diff = float(np.abs(actual - expected).max())
            checks.append(ChainCheck(label, actual.shape[-1], diff))
            if diff > tolerance:
                raise AssertionError(f"{label}: max |onnx - torch| = {diff} > {tolerance}")
    return checks
