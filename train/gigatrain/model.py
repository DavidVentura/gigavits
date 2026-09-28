"""Multilingual Piper VITS: per-language front ends feeding one speaker-, language- and
condition-conditioned back end.

Built from the vendored upstream Piper modules (gigatrain/vits, GPL-3.0-or-later). Parameter names
inside each front end mirror upstream (`enc_p.encoder.*`, `enc_p.proj.*`, `dp.*`) so weights map
one to one from Piper checkpoints and ONNX voices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import autocast, nn
from torch.nn import functional as F

from .config import ModelShape
from .vits import attentions, commons
from .vits.models import Generator, PosteriorEncoder, ResidualCouplingBlock, StochasticDurationPredictor


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.scale * grad, None


class TextEncoderBody(nn.Module):
    """Upstream TextEncoder without its embedding, which is shared across languages."""

    def __init__(self, shape: ModelShape):
        super().__init__()
        self.encoder = attentions.Encoder(
            shape.hidden_channels, shape.filter_channels, shape.n_heads, shape.n_layers, shape.kernel_size, shape.p_dropout
        )
        self.proj = nn.Conv1d(shape.hidden_channels, shape.inter_channels * 2, 1)
        self.out_channels = shape.inter_channels

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs


class FrontEnd(nn.Module):
    """One language's text encoder and stochastic duration predictor."""

    def __init__(self, shape: ModelShape):
        super().__init__()
        self.enc_p = TextEncoderBody(shape)
        self.dp = StochasticDurationPredictor(shape.hidden_channels, 192, 3, 0.5, 4, gin_channels=shape.gin_channels)


class Conditioning(nn.Module):
    """g = speaker + language + recording-condition embedding, [b, gin, 1]."""

    def __init__(self, n_speakers: int, n_languages: int, n_conditions: int, gin_channels: int):
        super().__init__()
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.emb_lang = nn.Embedding(n_languages, gin_channels)
        self.emb_cond = nn.Embedding(n_conditions, gin_channels)

    def forward(self, sid: torch.Tensor, lid: torch.Tensor, cid: torch.Tensor) -> torch.Tensor:
        return (self.emb_g(sid) + self.emb_lang(lid) + self.emb_cond(cid)).unsqueeze(-1)


class Adversary(nn.Module):
    """Condition and speaker classifiers on the time-averaged text-encoder output, behind gradient
    reversal: the heads learn to classify, the front ends learn to hide what the heads find."""

    def __init__(self, hidden_channels: int, classifier_hidden: int, n_conditions: int, n_speakers: int):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(hidden_channels, classifier_hidden), nn.ReLU(), nn.Linear(classifier_hidden, n_conditions)
        )
        self.speaker = nn.Sequential(
            nn.Linear(hidden_channels, classifier_hidden), nn.ReLU(), nn.Linear(classifier_hidden, n_speakers)
        )

    def forward(self, pooled: torch.Tensor, reversal_scale: float):
        reversed_ = GradientReversal.apply(pooled, reversal_scale)
        return self.condition(reversed_), self.speaker(reversed_)


def masked_mean(x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
    return (x * x_mask).sum(2) / x_mask.sum(2)


def language_speaker_mask(language_speakers: list[list[int]], n_speakers: int) -> torch.Tensor:
    """[n_languages, n_speakers] bool: which speakers the speaker classifier may choose between."""
    mask = torch.zeros(len(language_speakers), n_speakers, dtype=torch.bool)
    for lid, speakers in enumerate(language_speakers):
        mask[lid, speakers] = True
    return mask


def speaker_adversary_loss(logits: torch.Tensor, sid: torch.Tensor, lid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy among the speakers of each item's language. Items of single-speaker languages
    are skipped: there, knowing the speaker means knowing the language, and removing that would strip
    language information from the front end."""
    allowed = mask[lid]
    active = allowed.sum(1) > 1
    if not bool(active.any()):
        return logits.new_zeros(())
    restricted = logits.float().masked_fill(~allowed, float("-inf"))
    return F.cross_entropy(restricted[active], sid[active])


@dataclass
class TextEncoding:
    x: torch.Tensor
    m_p: torch.Tensor
    logs_p: torch.Tensor
    x_mask: torch.Tensor


@dataclass
class Latent:
    """Everything up to the expanded prior, shared by the back-end and front-end jobs."""

    text: TextEncoding
    g: torch.Tensor
    z: torch.Tensor
    z_p: torch.Tensor
    m_q: torch.Tensor
    logs_q: torch.Tensor
    y_mask: torch.Tensor
    attn: torch.Tensor
    m_p: torch.Tensor
    logs_p: torch.Tensor
    duration_loss: torch.Tensor


def _expand(attn: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    """[b, 1, t_y, t_x] x [b, d, t_x] -> [b, d, t_y]."""
    return torch.matmul(attn.squeeze(1), stats.transpose(1, 2)).transpose(1, 2)


class MultilingualVits(nn.Module):
    def __init__(
        self,
        shape: ModelShape,
        n_vocab: int,
        languages: tuple[str, ...],
        n_speakers: int,
        n_conditions: int,
        spec_channels: int,
    ):
        super().__init__()
        self.init_args = dict(
            shape=shape, n_vocab=n_vocab, languages=languages, n_speakers=n_speakers,
            n_conditions=n_conditions, spec_channels=spec_channels,
        )
        self.shape = shape
        self.languages = languages
        self.hidden_channels = shape.hidden_channels
        self.emb = nn.Embedding(n_vocab, shape.hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, shape.hidden_channels**-0.5)
        self.front_ends = nn.ModuleDict({lang: FrontEnd(shape) for lang in languages})
        self.cond = Conditioning(n_speakers, len(languages), n_conditions, shape.gin_channels)
        self.enc_q = PosteriorEncoder(
            spec_channels, shape.inter_channels, shape.hidden_channels, 5, 1, 16, gin_channels=shape.gin_channels
        )
        self.flow = ResidualCouplingBlock(
            shape.inter_channels, shape.hidden_channels, 5, 1, 4, gin_channels=shape.gin_channels
        )
        self.dec = make_decoder(shape)
        self.adversary = Adversary(shape.hidden_channels, shape.classifier_hidden, n_conditions, n_speakers)

    def front_end(self, lid: int) -> FrontEnd:
        return self.front_ends[self.languages[lid]]

    def embed(self, x: torch.Tensor, x_lengths: torch.Tensor):
        h = self.emb(x) * math.sqrt(self.hidden_channels)
        h = h.transpose(1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, h.size(2)), 1).type_as(h)
        return h, x_mask

    def _language_groups(self, lid: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
        present = torch.unique(lid).tolist()
        return [(language, (lid == language).nonzero(as_tuple=True)[0]) for language in present]

    def encode_text(self, x: torch.Tensor, x_lengths: torch.Tensor, lid: torch.Tensor) -> TextEncoding:
        """Each item goes through its own language's front end; results are scattered back in order."""
        h, x_mask = self.embed(x, x_lengths)
        groups = self._language_groups(lid)
        outputs = [(index, self.front_end(language).enc_p(h[index], x_mask[index])) for language, index in groups]
        hidden = h.new_zeros(h.shape).to(outputs[0][1][0].dtype)
        m_p = hidden.new_zeros(h.size(0), self.shape.inter_channels, h.size(2))
        logs_p = torch.zeros_like(m_p)
        for index, (x_l, m_l, logs_l) in outputs:
            hidden = hidden.index_copy(0, index, x_l)
            m_p = m_p.index_copy(0, index, m_l)
            logs_p = logs_p.index_copy(0, index, logs_l)
        return TextEncoding(hidden, m_p, logs_p, x_mask.to(hidden.dtype))

    def duration_nll(self, text: TextEncoding, w: torch.Tensor, g: torch.Tensor, lid: torch.Tensor) -> torch.Tensor:
        """Per-item negative log-likelihood of the durations w [b, 1, t_x] under each language's SDP."""
        nll = w.new_zeros(w.size(0), dtype=torch.float32)
        # The SDP's flows take logs and exps of durations; bf16 there produces NaNs.
        with autocast(w.device.type, enabled=False):
            for language, index in self._language_groups(lid):
                dp = self.front_end(language).dp
                nll_l = dp(text.x[index].float(), text.x_mask[index].float(), w[index].float(), g=g[index].float())
                nll = nll.index_copy(0, index, nll_l.float())
        return nll

    def latent(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
        sid: torch.Tensor,
        lid: torch.Tensor,
        cid: torch.Tensor,
        durations: torch.Tensor,
        has_durations: torch.Tensor,
    ) -> Latent:
        # Imported here so export and inference work without the compiled MAS kernel.
        from .alignment import alignment

        text = self.encode_text(x, x_lengths, lid)
        g = self.cond(sid, lid, cid)
        z, m_q, logs_q, y_mask = self.enc_q(spec, spec_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        attn = alignment(durations, has_durations, text.x_mask, y_mask, z_p, text.m_p, text.logs_p)
        w = attn.sum(2)
        duration_loss = self.duration_nll(text, w, g, lid).sum() / text.x_mask.float().sum()
        return Latent(
            text=text,
            g=g,
            z=z,
            z_p=z_p,
            m_q=m_q,
            logs_q=logs_q,
            y_mask=y_mask,
            attn=attn,
            m_p=_expand(attn.to(text.m_p.dtype), text.m_p),
            logs_p=_expand(attn.to(text.logs_p.dtype), text.logs_p),
            duration_loss=duration_loss,
        )

    def infer_prior(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        lid: torch.Tensor,
        cid: torch.Tensor,
        noise_scale: torch.Tensor | float,
        length_scale: torch.Tensor | float,
        noise_scale_w: torch.Tensor | float,
        front_end: FrontEnd,
    ):
        """Front-end inference for one language: -> (z_p, y_mask, durations [b, t_x])."""
        h, x_mask = self.embed(x, x_lengths)
        hidden, m_p, logs_p = front_end.enc_p(h, x_mask)
        g = self.cond(sid, lid, cid)
        logw = front_end.dp(hidden, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)
        m_p = _expand(attn, m_p)
        logs_p = _expand(attn, logs_p)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        return z_p, y_mask, w_ceil.squeeze(1).long()

    def infer_latent(self, z_p: torch.Tensor, y_mask: torch.Tensor, sid, lid, cid) -> torch.Tensor:
        g = self.cond(sid, lid, cid)
        return self.flow(z_p, y_mask, g=g, reverse=True) * y_mask

    def infer_audio(self, z: torch.Tensor, sid, lid, cid, decoder: Generator) -> torch.Tensor:
        return decoder(z, g=self.cond(sid, lid, cid))

    def infer(self, x, x_lengths, sid, lid, cid, noise_scale, length_scale, noise_scale_w, decoder: Generator):
        """Full PyTorch reference path; lid picks the front end, so the batch must be one language."""
        language = int(lid[0])
        if bool((lid != language).any()):
            raise ValueError("infer takes one language per batch")
        z_p, y_mask, _ = self.infer_prior(
            x, x_lengths, sid, lid, cid, noise_scale, length_scale, noise_scale_w, self.front_end(language)
        )
        z = self.infer_latent(z_p, y_mask, sid, lid, cid)
        return self.infer_audio(z, sid, lid, cid, decoder)


def make_decoder(shape: ModelShape) -> Generator:
    return Generator(
        shape.inter_channels,
        shape.resblock,
        shape.resblock_kernel_sizes,
        shape.resblock_dilation_sizes,
        shape.upsample_rates,
        shape.upsample_initial_channel,
        shape.upsample_kernel_sizes,
        gin_channels=shape.gin_channels,
    )


def copy_model(model: MultilingualVits) -> MultilingualVits:
    # deepcopy fails on weight-normed modules once they have run a forward pass.
    clone = MultilingualVits(**model.init_args)
    clone.load_state_dict(model.state_dict())
    return clone


def copy_decoder(decoder: Generator, shape: ModelShape) -> Generator:
    clone = make_decoder(shape)
    clone.load_state_dict(decoder.state_dict())
    return clone
