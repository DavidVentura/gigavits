import numpy as np
import pytest

from gigatrain.sampling import BucketedBatchSampler, plan_epoch, sampling_plan, sampling_probabilities


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


def uniform_plan(languages):
    return sampling_plan(languages, np.ones(len(languages)), np.ones(len(languages)))


def test_one_language_per_batch_and_buckets_by_length():
    rng = np.random.default_rng(0)
    languages = np.arange(600) % 6
    lengths = rng.integers(1, 1000, 600)
    batches = plan_epoch(uniform_plan(languages), lengths, 8, 60, 10, 1, rng)
    assert len(batches) == 60 and all(len(b) == 8 for b in batches)
    assert all(len(set(languages[b])) == 1 for b in batches)
    spread = np.mean([np.ptp(lengths[b]) for b in batches])
    unsorted = np.mean([np.ptp(lengths[rng.choice(600, 8)]) for _ in range(200)])
    assert spread < unsorted / 2


def test_k_languages_per_batch_split_evenly():
    languages = np.arange(300) % 5
    batches = plan_epoch(uniform_plan(languages), np.arange(300), 10, 40, 8, 3, np.random.default_rng(1))
    for b in batches:
        counts = sorted(np.bincount(languages[b], minlength=5))
        assert counts == [0, 0, 3, 3, 4]


def test_frequencies_follow_language_mass_then_weight():
    # en: 4 h (item weights 3:1), es: 1 h.
    plan = sampling_plan(["en", "en", "es"], [3.0, 1.0, 2.0], [7200.0, 7200.0, 3600.0])
    batches = plan_epoch(plan, np.array([1, 2, 3]), 10, 3000, 4, 1, np.random.default_rng(2))
    counts = np.bincount(np.concatenate(batches), minlength=3) / 30000
    assert counts == pytest.approx(plan.item_probabilities(3), abs=0.015)


def test_too_many_languages_per_batch_raises():
    with pytest.raises(ValueError):
        plan_epoch(uniform_plan(np.array([0, 1])), np.array([1, 2]), 4, 1, 1, 3, np.random.default_rng(0))


def sampler(rank=0, world=1, seed=5, k=2):
    return BucketedBatchSampler(uniform_plan(np.arange(50) % 3), np.arange(50), 4, 6, 2, k, seed, rank, world)


def test_sampler_is_a_function_of_seed_and_epoch():
    a, b = sampler(), sampler()
    a.set_epoch(3)
    b.set_epoch(3)
    assert list(a) == list(b)
    b.set_epoch(4)
    assert list(a) != list(b)


def test_ranks_take_disjoint_batches_of_one_plan():
    r0, r1, whole = sampler(0, 2), sampler(1, 2), BucketedBatchSampler(
        uniform_plan(np.arange(50) % 3), np.arange(50), 4, 12, 2, 2, 5, 0, 1
    )
    assert len(list(r0)) == len(list(r1)) == 6
    assert list(r0) + list(r1) != [] and sorted(map(tuple, list(r0) + list(r1))) == sorted(map(tuple, list(whole)))
