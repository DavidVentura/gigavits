import pytest
import torch

from gigatrain.config import ScheduleConfig
from gigatrain.model import GradientReversal, language_speaker_mask, speaker_adversary_loss
from gigatrain.training import Phase, adversarial_scale, phase_at

# Speakers 0, 1 speak language 0; speaker 2 alone speaks language 1; speaker 1 also speaks language 2 with 3.
MASK = language_speaker_mask([[0, 1], [2], [1, 3]], 4)


def test_mask_rows():
    assert MASK.tolist() == [
        [True, True, False, False],
        [False, False, True, False],
        [False, True, False, True],
    ]


def test_single_speaker_language_is_skipped():
    logits = torch.randn(2, 4, requires_grad=True)
    loss = speaker_adversary_loss(logits, torch.tensor([2, 2]), torch.tensor([1, 1]), MASK)
    assert loss.item() == 0.0


def test_loss_only_competes_within_the_language():
    logits = torch.zeros(1, 4, requires_grad=True)
    loss = speaker_adversary_loss(logits, torch.tensor([1]), torch.tensor([0]), MASK)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    loss.backward()
    assert logits.grad[0, 2:].abs().sum() == 0
    # Raising another language's speaker changes nothing.
    boosted = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
    assert speaker_adversary_loss(boosted, torch.tensor([1]), torch.tensor([0]), MASK).item() == pytest.approx(loss.item())


def test_mixed_batch_averages_over_active_items_only():
    logits = torch.zeros(3, 4)
    loss = speaker_adversary_loss(logits, torch.tensor([0, 2, 3]), torch.tensor([0, 1, 2]), MASK)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_gradient_reversal_scales_and_flips():
    x = torch.ones(3, requires_grad=True)
    GradientReversal.apply(x, 0.25).sum().backward()
    assert x.grad.tolist() == [-0.25] * 3


def test_ramp_and_phase():
    s = ScheduleConfig(gan_start_step=30_000, adversarial_ramp_steps=20_000, adversarial_weight=0.02)
    assert adversarial_scale(0, s) == 0.0
    assert adversarial_scale(10_000, s) == pytest.approx(0.01)
    assert adversarial_scale(50_000, s) == pytest.approx(0.02)
    assert phase_at(29_999, s) is Phase.RECONSTRUCTION
    assert phase_at(30_000, s) is Phase.ADVERSARIAL
