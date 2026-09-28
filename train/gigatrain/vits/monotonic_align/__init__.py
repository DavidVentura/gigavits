# Derived from piper1-gpl src/piper/train/vits/monotonic_align/__init__.py
# at commit efffbfb226bfb511ebbcf55d0cecd8b35a89743d. GPL-3.0-or-later, see train/COPYING.

import numpy as np
import torch

from .core import maximum_path_c


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """neg_cent, mask: [b, t_t, t_s]. Runs on CPU; returns the path on neg_cent's device."""
    device = neg_cent.device
    dtype = neg_cent.dtype
    values = neg_cent.detach().float().cpu().numpy()
    path = np.zeros(values.shape, dtype=np.int32)
    t_t_max = mask.sum(1)[:, 0].detach().cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].detach().cpu().numpy().astype(np.int32)
    maximum_path_c(path, np.ascontiguousarray(values), t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
