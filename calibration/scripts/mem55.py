"""GPU memory for a VITS medium model with N per-language front ends (TextEncoder + duration predictor).

A batch mixes languages: each sample is routed through its language's front end, the rest of the
generator (posterior encoder, flow, decoder) and the discriminators are shared. All front ends are
primed with a gradient first so AdamW holds state for every one of them (steady state of a long run).

Usage: mem55.py <num_front_ends> <batch_size> <precision: fp32|bf16>
"""

import json
import math
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from piper.train.vits import commons
from piper.train.vits.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from piper.train.vits.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from piper.train.vits.models import MultiPeriodDiscriminator, StochasticDurationPredictor, SynthesizerTrn, TextEncoder

SAMPLE_RATE = 22050
HOP = 256
N_FFT = 1024
SEGMENT = 8192
HIDDEN = 192
GIN = 512
NUM_SPEAKERS = 64


class RoutedTextEncoder(nn.Module):
    """Batch must be sorted by language; `groups` holds (language, count) runs."""

    def __init__(self, encoders: nn.ModuleList) -> None:
        super().__init__()
        self.encoders = encoders
        self.groups: list[tuple[int, int]] = []

    def forward(self, x, x_lengths):
        outs, start = [], 0
        for lang, count in self.groups:
            outs.append(self.encoders[lang](x[start : start + count], x_lengths[start : start + count]))
            start += count
        return tuple(torch.cat(parts, dim=0) for parts in zip(*outs))


class RoutedDurationPredictor(nn.Module):
    def __init__(self, predictors: nn.ModuleList, encoder: RoutedTextEncoder) -> None:
        super().__init__()
        self.predictors = predictors
        self.encoder = encoder

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        outs, start = [], 0
        for lang, count in self.encoder.groups:
            sl = slice(start, start + count)
            outs.append(self.predictors[lang](x[sl], x_mask[sl], w[sl], g=g[sl]))
            start += count
        return torch.cat(outs, dim=0)


def millions(module: nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def main() -> None:
    num_fronts, batch_size, precision = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    dtype = {"fp32": None, "bf16": torch.bfloat16}[precision]
    torch.manual_seed(0)
    device = torch.device("cuda")

    model_g = SynthesizerTrn(
        n_vocab=256, spec_channels=N_FFT // 2 + 1, segment_size=SEGMENT // HOP, inter_channels=HIDDEN,
        hidden_channels=HIDDEN, filter_channels=768, n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1,
        resblock="2", resblock_kernel_sizes=(3, 5, 7), resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
        upsample_rates=(8, 8, 4), upsample_initial_channel=256, upsample_kernel_sizes=(16, 16, 8),
        n_speakers=NUM_SPEAKERS, gin_channels=GIN, use_sdp=True,
    )
    shared_g = millions(model_g) - millions(model_g.enc_p) - millions(model_g.dp)
    per_front_enc, per_front_dp = millions(model_g.enc_p), millions(model_g.dp)
    encoders = nn.ModuleList(
        [TextEncoder(256, HIDDEN, HIDDEN, 768, 2, 6, 3, 0.1) for _ in range(num_fronts)]
    )
    predictors = nn.ModuleList(
        [StochasticDurationPredictor(HIDDEN, 192, 3, 0.5, 4, gin_channels=GIN) for _ in range(num_fronts)]
    )
    model_g.enc_p = RoutedTextEncoder(encoders)
    model_g.dp = RoutedDurationPredictor(predictors, model_g.enc_p)
    model_d = MultiPeriodDiscriminator()
    model_g.to(device)
    model_d.to(device)
    opt_g = torch.optim.AdamW(model_g.parameters(), lr=2e-4, betas=(0.8, 0.99), eps=1e-9)
    opt_d = torch.optim.AdamW(model_d.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-9)
    after_build = torch.cuda.memory_allocated() / 1e9

    # Prime: one gradient into every front end so AdamW allocates state for all of them.
    x = torch.randint(1, 256, (1, 16), device=device)
    xl = torch.tensor([16], device=device)
    g = torch.zeros(1, GIN, 1, device=device)
    prime = 0.0
    for enc, dp in zip(encoders, predictors):
        h, m, logs, mask = enc(x, xl)
        prime = prime + m.mean() + logs.mean() + dp(h, mask, torch.ones_like(mask), g=g).mean()
    prime.backward()
    opt_g.step()
    opt_g.zero_grad(set_to_none=True)
    after_prime = torch.cuda.memory_allocated() / 1e9

    torch.cuda.reset_peak_memory_stats()
    timings = []
    for step in range(3):
        langs = sorted(torch.randint(0, num_fronts, (batch_size,)).tolist())
        groups: list[tuple[int, int]] = []
        for lang in langs:
            if groups and groups[-1][0] == lang:
                groups[-1] = (lang, groups[-1][1] + 1)
            else:
                groups.append((lang, 1))
        model_g.enc_p.groups = groups
        spec_frames = torch.randint(300, 900, (batch_size,), device=device).sort(descending=True).values
        phonemes = (spec_frames.float() / 4.5).long()
        t_x, t_y = int(phonemes.max()), int(spec_frames.max())
        x = torch.randint(1, 256, (batch_size, t_x), device=device)
        spec = torch.rand(batch_size, N_FFT // 2 + 1, t_y, device=device)
        y = torch.randn(batch_size, 1, t_y * HOP, device=device) * 0.1
        sid = torch.randint(0, NUM_SPEAKERS, (batch_size,), device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
            y_hat, l_length, _, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = model_g(x, phonemes, spec, spec_frames, sid)
            with torch.autocast("cuda", enabled=False):
                mel = spec_to_mel_torch(spec, N_FFT, 80, SAMPLE_RATE, 0.0, None)
                y_mel = commons.slice_segments(mel, ids_slice, SEGMENT // HOP)
                y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float(), N_FFT, 80, SAMPLE_RATE, HOP, N_FFT, 0.0, None)
            y_seg = commons.slice_segments(y, ids_slice * HOP, SEGMENT)
            _, y_d_hat_g, fmap_r, fmap_g = model_d(y_seg, y_hat)
            with torch.autocast("cuda", enabled=False):
                loss_gen, _ = generator_loss(y_d_hat_g)
                loss_g = (
                    loss_gen + feature_loss(fmap_r, fmap_g) + 45 * F.l1_loss(y_mel, y_hat_mel)
                    + torch.sum(l_length.float()) + kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                )
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
            y_d_hat_r, y_d_hat_g, _, _ = model_d(y_seg, y_hat.detach())
            with torch.autocast("cuda", enabled=False):
                loss_d, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
        assert math.isfinite(float(loss_g)) and math.isfinite(float(loss_d))

    result = {
        "num_front_ends": num_fronts,
        "batch_size": batch_size,
        "precision": precision,
        "params_per_front_end_m": round(per_front_enc + per_front_dp, 3),
        "params_text_encoder_m": round(per_front_enc, 3),
        "params_duration_predictor_sdp_m": round(per_front_dp, 3),
        "params_all_front_ends_m": round(millions(encoders) + millions(predictors), 1),
        "params_shared_generator_m": round(shared_g, 2),
        "params_discriminator_m": round(millions(model_d), 2),
        "params_total_m": round(millions(model_g) + millions(model_d), 1),
        "gpu_alloc_after_build_gb": round(after_build, 2),
        "gpu_alloc_after_adam_state_gb": round(after_prime, 2),
        "gpu_peak_alloc_train_step_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "step_seconds_last": round(timings[-1], 3),
        "distinct_front_ends_in_last_batch": len(groups),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
