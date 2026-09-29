"""Choosing between teacher durations and monotonic alignment search, per batch item."""
from __future__ import annotations

import math

import torch
from torch import autocast

from .vits.commons import generate_path
from .vits.monotonic_align import maximum_path


def neg_cross_entropy(z_p: torch.Tensor, m_p: torch.Tensor, logs_p: torch.Tensor) -> torch.Tensor:
    """Log-likelihood of every latent frame under every phoneme's prior, [b, t_y, t_x] (VITS)."""
    with autocast(z_p.device.type, enabled=False):
        z_p, m_p, logs_p = z_p.float(), m_p.float(), logs_p.float()
        s_p_sq_r = torch.exp(-2 * logs_p)
        neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
        neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
        neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
        neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
        return neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4


def teacher_path(durations: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """durations [b, t_x] frames per phoneme id, attn_mask [b, 1, t_y, t_x] -> path [b, 1, t_y, t_x]."""
    return generate_path(durations.unsqueeze(1).to(attn_mask.dtype), attn_mask)


def alignment(
    durations: torch.Tensor,
    teacher_rows: torch.Tensor,
    searched_rows: torch.Tensor,
    x_mask: torch.Tensor,
    y_mask: torch.Tensor,
    z_p: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
) -> torch.Tensor:
    """Hard alignment [b, 1, t_y, t_x]: teacher durations for teacher_rows, MAS for searched_rows.

    MAS runs on CPU and only for the rows that need it; a batch without such rows never leaves the
    device. Row counts are tensor shapes, so checking them does not sync.
    """
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = torch.zeros_like(attn_mask)
    if teacher_rows.numel():
        attn[teacher_rows] = teacher_path(durations[teacher_rows], attn_mask[teacher_rows])
    if searched_rows.numel():
        with torch.no_grad():
            neg_cent = neg_cross_entropy(z_p[searched_rows], m_p[searched_rows], logs_p[searched_rows])
            path = maximum_path(neg_cent, attn_mask[searched_rows].squeeze(1))
        attn[searched_rows] = path.unsqueeze(1).to(attn.dtype)
    return attn.detach()
