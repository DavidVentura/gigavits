import torch

from gigatrain.alignment import alignment, teacher_path
from gigatrain.vits.commons import sequence_mask


def masks(x_lengths, y_lengths):
    x_mask = sequence_mask(torch.tensor(x_lengths), max(x_lengths)).unsqueeze(1).float()
    y_mask = sequence_mask(torch.tensor(y_lengths), max(y_lengths)).unsqueeze(1).float()
    return x_mask, y_mask


def test_teacher_path_expands_each_phoneme_by_its_duration():
    durations = torch.tensor([[2, 0, 3, 1]])
    x_mask, y_mask = masks([4], [6])
    attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
    path = teacher_path(durations, attn_mask)
    assert path.shape == (1, 1, 6, 4)
    assert path.sum(2).squeeze().tolist() == [2, 0, 3, 1]
    assert path[0, 0].argmax(1).tolist() == [0, 0, 2, 2, 2, 3]


def test_mixed_batch_uses_durations_where_present_and_mas_elsewhere():
    torch.manual_seed(0)
    x_mask, y_mask = masks([4, 3], [8, 7])
    durations = torch.tensor([[1, 1, 1, 5], [0, 0, 0, 0]])
    has = torch.tensor([True, False])
    channels = 5
    z_p = torch.randn(2, channels, 8)
    # Make item 0's prior favour a different alignment than its teacher durations.
    m_p = torch.randn(2, channels, 4)
    logs_p = torch.zeros(2, channels, 4)
    attn = alignment(durations, has, x_mask, y_mask, z_p, m_p, logs_p)
    w = attn.sum(2).squeeze(1)
    assert w[0].tolist() == [1, 1, 1, 5]
    # MAS gives every phoneme at least one frame and covers every frame once, monotonically.
    assert (w[1, :3] >= 1).all() and w[1, 3] == 0 and w[1].sum() == 7
    frames_to_phoneme = attn[1, 0, :7].argmax(1)
    assert (frames_to_phoneme.diff() >= 0).all()
    assert attn[1, 0, 7].sum() == 0


def test_all_teacher_batch_never_needs_the_prior():
    x_mask, y_mask = masks([2], [3])
    nan = torch.full((1, 2, 2), float("nan"))
    attn = alignment(torch.tensor([[1, 2]]), torch.tensor([True]), x_mask, y_mask, torch.zeros(1, 2, 3), nan, nan)
    assert attn.sum(2).squeeze().tolist() == [1, 2]
