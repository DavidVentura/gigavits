import numpy as np
import pytest

from gigatrain.sampling import BucketedBatchSampler, plan_epoch, sampling_probabilities


def test_languages_get_sqrt_hours_and_items_share_by_weight():
    # en: 4 h over two items (weights 3:1), es: 1 h in one item.
    p = sampling_probabilities(["en", "en", "es"], [3.0, 1.0, 2.0], [7200.0, 7200.0, 3600.0])
    assert p.sum() == pytest.approx(1.0)
    en, es = p[0] + p[1], p[2]
    assert en / es == pytest.approx(np.sqrt(4.0) / np.sqrt(1.0))
    assert p[0] / p[1] == pytest.approx(3.0)


def test_item_weight_does_not_move_language_mass():
    a = sampling_probabilities(["en", "es"], [1.0, 1.0], [100.0, 100.0])
    b = sampling_probabilities(["en", "es"], [10.0, 1.0], [100.0, 100.0])
    assert a == pytest.approx(b)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        sampling_probabilities(["en"], [0.0], [1.0])
    with pytest.raises(ValueError):
        sampling_probabilities(["en"], [1.0, 2.0], [1.0])


def test_plan_epoch_buckets_by_length():
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 1000, 500)
    p = np.full(500, 1 / 500)
    batches = plan_epoch(p, lengths, batch_size=8, num_batches=40, bucket_batches=10, rng=rng)
    assert len(batches) == 40 and all(len(b) == 8 for b in batches)
    spread = np.mean([np.ptp(lengths[b]) for b in batches])
    unsorted = np.mean([np.ptp(lengths[rng.choice(500, 8)]) for _ in range(200)])
    assert spread < unsorted / 3


def test_plan_follows_probabilities():
    rng = np.random.default_rng(1)
    p = np.array([0.7, 0.2, 0.1])
    batches = plan_epoch(p, np.array([1, 2, 3]), batch_size=10, num_batches=2000, bucket_batches=4, rng=rng)
    counts = np.bincount(np.concatenate(batches), minlength=3) / 20000
    assert counts == pytest.approx(p, abs=0.01)


def sampler(rank=0, world=1, seed=5):
    return BucketedBatchSampler(np.full(50, 0.02), np.arange(50), 4, 6, 2, seed, rank, world)


def test_sampler_is_a_function_of_seed_and_epoch():
    a, b = sampler(), sampler()
    a.set_epoch(3)
    b.set_epoch(3)
    assert list(a) == list(b)
    b.set_epoch(4)
    assert list(a) != list(b)


def test_ranks_take_disjoint_batches_of_one_plan():
    r0, r1 = sampler(0, 2), sampler(1, 2)
    assert len(list(r0)) == len(list(r1)) == 6
    assert list(r0) != list(r1)
