import numpy as np
import pytest
import torch

from gigatrain.extend import ExtendError, check_extension, grow_rows, grown_optimizer_state, grown_state
from gigatrain.ids import IdMaps
from gigatrain.records import Condition

OLD = IdMaps(("a", "b"), ("en", "es"), (("en", "en"), ("es", "es")), (("en", "a"), ("es", "b")))
NEW = IdMaps(("a", "b", "c"), ("en", "es", "pl"), (("en", "en"), ("es", "es"), ("pl", "pl")),
             (("en", "a"), ("es", "b"), ("pl", "c")))


def test_grow_rows_keeps_old_rows_and_fresh_new_ones():
    grown = grow_rows(torch.ones(2, 3), torch.zeros(4, 3), "k")
    assert grown.sum(1).tolist() == [3, 3, 0, 0]
    with pytest.raises(ExtendError):
        grow_rows(torch.ones(2, 3), torch.zeros(4, 2), "k")
    with pytest.raises(ExtendError):
        grow_rows(torch.ones(4, 3), torch.zeros(2, 3), "k")


def test_extension_must_append_and_name_a_closest_language():
    check_extension(OLD, NEW, {"pl": "es"})
    with pytest.raises(ExtendError, match="renumbered"):
        check_extension(OLD, IdMaps(("b", "a"), OLD.languages, OLD.language_espeak, OLD.language_speakers), {})
    with pytest.raises(ExtendError, match="closest"):
        check_extension(OLD, NEW, {})
    with pytest.raises(ExtendError, match="existing"):
        check_extension(OLD, NEW, {"pl": "de"})
    reordered = IdMaps(OLD.speakers, OLD.languages, OLD.language_espeak, OLD.language_speakers,
                       (Condition.DEGRADED, Condition.CLEAN, Condition.NARROW_BAND))
    with pytest.raises(ExtendError, match="conditions"):
        check_extension(OLD, reordered, {})


def fake_states():
    old = {
        "model.cond.emb_lang.weight": torch.tensor([[1.0], [2.0]]),
        "model.cond.emb_g.weight": torch.tensor([[5.0], [6.0]]),
        "model.front_ends.es.enc_p.proj.bias": torch.tensor([7.0]),
        "model.front_ends.es.dp.post_pre.bias": torch.tensor([8.0]),
        "model.flow.x": torch.tensor([9.0]),
    }
    fresh = {
        "model.cond.emb_lang.weight": torch.zeros(3, 1),
        "model.cond.emb_g.weight": torch.full((3, 1), -1.0),
        "model.front_ends.es.enc_p.proj.bias": torch.zeros(1),
        "model.front_ends.es.dp.post_pre.bias": torch.zeros(1),
        "model.front_ends.pl.enc_p.proj.bias": torch.zeros(1),
        "model.front_ends.pl.dp.post_pre.bias": torch.zeros(1),
        "model.flow.x": torch.zeros(1),
    }
    return old, fresh


def test_grown_state_copies_like_language_and_takes_voice_weights():
    old, fresh = fake_states()
    out = grown_state(old, fresh, OLD, NEW, {"pl": "es"}, {})
    assert out["model.front_ends.pl.enc_p.proj.bias"].item() == 7.0
    assert out["model.cond.emb_lang.weight"].squeeze().tolist() == [1.0, 2.0, 2.0]
    assert out["model.cond.emb_g.weight"].squeeze().tolist() == [5.0, 6.0, -1.0]
    assert out["model.flow.x"].item() == 9.0
    voiced = grown_state(old, fresh, OLD, NEW, {"pl": "es"}, {"pl": {"enc_p.proj.bias": np.array([3.0], dtype=np.float32)}})
    assert voiced["model.front_ends.pl.enc_p.proj.bias"].item() == 3.0
    # Training-only duration-predictor parts are not in voices; they stay with the closest language.
    assert voiced["model.front_ends.pl.dp.post_pre.bias"].item() == 8.0


def test_grown_state_rejects_unexpected_changes():
    old, fresh = fake_states()
    with pytest.raises(ExtendError, match="new but"):
        grown_state(old, fresh | {"model.flow.y": torch.zeros(1)}, OLD, NEW, {"pl": "es"}, {})
    with pytest.raises(ExtendError, match="shape"):
        grown_state(old, fresh | {"model.flow.x": torch.zeros(2)}, OLD, NEW, {"pl": "es"}, {})
    with pytest.raises(ExtendError, match="has no"):
        grown_state(old, fresh, OLD, NEW, {"pl": "es"}, {"pl": {}})


def test_optimizer_state_follows_names_and_pads_grown_rows():
    old = {
        "state": {
            0: {"step": torch.tensor(3.0), "exp_avg": torch.ones(2, 1), "exp_avg_sq": torch.ones(2, 1)},
            1: {"step": torch.tensor(3.0), "exp_avg": torch.full((1,), 2.0), "exp_avg_sq": torch.full((1,), 2.0)},
        },
        "param_groups": [{"lr": 1e-4, "params": [0, 1]}],
    }
    names = ["model.cond.emb_g.weight", "model.flow.x"]
    new_names = ["model.front_ends.pl.w", "model.cond.emb_g.weight", "model.flow.x"]
    shapes = {"model.front_ends.pl.w": torch.Size([1]), "model.cond.emb_g.weight": torch.Size([3, 1]), "model.flow.x": torch.Size([1])}
    out = grown_optimizer_state(old, names, new_names, shapes)
    assert out["param_groups"][0]["params"] == [0, 1, 2] and out["param_groups"][0]["lr"] == 1e-4
    assert 0 not in out["state"]
    assert out["state"][1]["exp_avg"].squeeze().tolist() == [1.0, 1.0, 0.0]
    assert out["state"][2]["exp_avg"].item() == 2.0
    with pytest.raises(ExtendError, match="changed shape"):
        grown_optimizer_state(old, names, new_names, shapes | {"model.flow.x": torch.Size([2])})
