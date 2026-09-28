import numpy as np
import pytest
import torch

from conftest import AMY_ONNX, TOKEN_TABLE
from gigatrain.config import DataConfig, ModelShape, TrainConfig, to_jsonable
from gigatrain.ids import IdMaps
from gigatrain.init_from_piper import (
    FromCheckpoint,
    FromVoice,
    Fresh,
    InitError,
    apply_plan,
    init_plan,
    initial_state,
    random_piper_checkpoint_state,
    read_voice_front_end,
    source_for,
)
from gigatrain.records import Vocab
from gigatrain.training import BackEndJob

VOCAB = Vocab.load(TOKEN_TABLE)

needs_voice = pytest.mark.skipif(
    not AMY_ONNX.exists(),
    reason=f"voice ONNX {AMY_ONNX} not found; set GIGAPIPER_TEST_VOICE_ONNX to en_US-amy-medium.onnx",
)


def test_rules():
    assert source_for("model.emb.weight", 175) == FromCheckpoint("model_g.enc_p.emb.weight", rows=175)
    assert source_for("model.front_ends.es-419.enc_p.encoder.attn_layers.0.conv_q.weight", 175) == FromVoice(
        "es-419", "enc_p.encoder.attn_layers.0.conv_q.weight"
    )
    assert source_for("model.front_ends.es.dp.flows.7.proj.bias", 175) == FromVoice("es", "dp.flows.7.proj.bias")
    assert source_for("model.front_ends.es.dp.flows.1.proj.bias", 175) == FromCheckpoint("model_g.dp.flows.1.proj.bias")
    assert source_for("model.front_ends.es.dp.post_pre.weight", 175) == FromCheckpoint("model_g.dp.post_pre.weight")
    assert source_for("model.front_ends.es.dp.cond.weight", 175) is Fresh.ZERO
    assert source_for("model.flow.flows.2.enc.cond_layer.weight_g", 175) is Fresh.ZERO
    assert source_for("model.flow.flows.2.enc.cond_layer.weight_v", 175) is Fresh.KEEP
    assert source_for("model.flow.flows.2.enc.in_layers.0.weight_v", 175) == FromCheckpoint(
        "model_g.flow.flows.2.enc.in_layers.0.weight_v"
    )
    assert source_for("model.dec.cond.bias", 175) is Fresh.ZERO
    assert source_for("model.enc_q.enc.cond_layer.bias", 175) is Fresh.ZERO
    assert source_for("model.cond.emb_lang.weight", 175) is Fresh.KEEP
    assert source_for("disc.discriminators.0.convs.0.weight_g", 175) == FromCheckpoint(
        "model_d.discriminators.0.convs.0.weight_g"
    )
    with pytest.raises(InitError):
        source_for("model.something_new.weight", 175)


def test_apply_plan_raises_on_shape_mismatch_missing_and_leftovers():
    target = {"model.dec.conv_pre.weight": torch.zeros(2, 3)}
    plan = init_plan(list(target), 175)
    with pytest.raises(InitError, match="shape"):
        apply_plan(plan, target, {"model_g.dec.conv_pre.weight": torch.zeros(3, 2)}, {})
    with pytest.raises(InitError, match="no model_g"):
        apply_plan(plan, target, {}, {})
    with pytest.raises(InitError, match="without a destination"):
        apply_plan(plan, target, {"model_g.dec.conv_pre.weight": torch.ones(2, 3), "model_g.flow.x": torch.ones(1)}, {})
    out = apply_plan(plan, target, {"model_g.dec.conv_pre.weight": torch.ones(2, 3), "model_g.enc_p.proj.bias": torch.ones(1)}, {})
    assert out["model.dec.conv_pre.weight"].sum() == 6


@needs_voice
def test_voice_reader_recovers_the_folded_affine_log_scale():
    weights = read_voice_front_end(AMY_ONNX)
    assert "enc_p.emb.weight" not in weights
    assert weights["dp.flows.0.logs"].shape == (2, 1)
    # amy's graph holds exp(-logs) = [2.131433, 1.053098].
    assert np.exp(-weights["dp.flows.0.logs"]).ravel() == pytest.approx([2.131433, 1.053098], rel=1e-6)
    assert not any(name.startswith(("dp.post", "dp.flows.1.")) for name in weights)


@pytest.fixture(scope="module")
def initialised():
    config = TrainConfig(data=DataConfig(shards=(), token_table=str(TOKEN_TABLE)))
    ids = IdMaps(
        speakers=("a", "b", "c"),
        languages=("en-us", "es"),
        language_espeak=(("en-us", "en-us"), ("es", "es")),
        language_speakers=(("en-us", "a"), ("en-us", "b"), ("es", "c")),
    )
    job = BackEndJob(to_jsonable(config), ids.to_json(), VOCAB.size)
    base = random_piper_checkpoint_state(ModelShape())
    state = initial_state(job.state_dict(), base, {"en-us": AMY_ONNX, "es": AMY_ONNX}, VOCAB.size)
    job.load_state_dict(state, strict=True)
    return job, base


@needs_voice
def test_initialised_model_takes_every_source(initialised):
    job, base = initialised
    amy = read_voice_front_end(AMY_ONNX)
    fe = job.model.front_ends["es"]
    assert torch.equal(fe.enc_p.encoder.ffn_layers[3].conv_1.weight, torch.from_numpy(amy["enc_p.encoder.ffn_layers.3.conv_1.weight"]))
    assert torch.equal(fe.dp.flows[0].logs, torch.from_numpy(amy["dp.flows.0.logs"]))
    assert torch.equal(fe.dp.post_pre.weight, base["model_g.dp.post_pre.weight"])
    assert torch.equal(job.model.emb.weight, base["model_g.enc_p.emb.weight"][:175])
    assert torch.equal(job.model.flow.flows[0].enc.in_layers[0].weight_v, base["model_g.flow.flows.0.enc.in_layers.0.weight_v"])
    assert torch.equal(job.disc.discriminators[3].convs[1].weight_v, base["model_d.discriminators.3.convs.1.weight_v"])
    assert fe.dp.cond.weight.abs().sum() == 0
    assert job.model.dec.cond.weight.abs().sum() == 0


@needs_voice
def test_step_zero_back_end_ignores_conditioning(initialised):
    """Zeroed conditioning layers make flow and decoder reproduce the single-speaker checkpoint for any g."""
    from gigatrain.vits.models import Generator, ResidualCouplingBlock

    job, base = initialised
    shape = ModelShape()
    flow = ResidualCouplingBlock(shape.inter_channels, shape.hidden_channels, 5, 1, 4, gin_channels=0)
    flow.load_state_dict({k.removeprefix("model_g.flow."): v for k, v in base.items() if k.startswith("model_g.flow.")})
    dec = Generator(shape.inter_channels, shape.resblock, shape.resblock_kernel_sizes, shape.resblock_dilation_sizes,
                    shape.upsample_rates, shape.upsample_initial_channel, shape.upsample_kernel_sizes, gin_channels=0)
    dec.load_state_dict({k.removeprefix("model_g.dec."): v for k, v in base.items() if k.startswith("model_g.dec.")})
    torch.manual_seed(0)
    z = torch.randn(1, shape.inter_channels, 20)
    mask = torch.ones(1, 1, 20)
    g = job.model.cond(torch.tensor([1]), torch.tensor([1]), torch.tensor([2]))
    with torch.no_grad():
        assert torch.allclose(job.model.flow(z, mask, g=g), flow(z, mask), atol=1e-5)
        assert torch.allclose(job.model.dec(z, g=g), dec(z), atol=1e-5)
