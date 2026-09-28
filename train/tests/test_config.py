import pytest

from gigatrain.config import ConfigError, Precision, TrainConfig, parse_dataclass, to_jsonable

DATA = {"shards": ["/s"], "token_table": "/t.json"}


def test_defaults_are_piper_medium_and_round_trip():
    config = parse_dataclass(TrainConfig, {"data": DATA})
    assert config.model.hidden_channels == 192 and config.model.gin_channels == 512
    assert config.precision is Precision.BF16_MIXED
    assert parse_dataclass(TrainConfig, to_jsonable(config)) == config


@pytest.mark.parametrize(
    "raw",
    [
        {"data": DATA, "unknown": 1},
        {"data": {"shards": ["/s"]}},
        {"data": DATA, "precision": "fp8"},
        {"data": DATA, "model": {"upsample_rates": [8, 8, 2]}},
        {"data": DATA | {"batch_size": "32"}},
        {"data": DATA, "audio": {"segment_size": 1000}},
    ],
)
def test_invalid_config_raises(raw):
    with pytest.raises(ConfigError):
        parse_dataclass(TrainConfig, raw)
