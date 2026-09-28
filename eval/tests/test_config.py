import pytest

from config import ConfigError, Hub, Local, RandomInit, load_config, parse_config
from repo import EVAL


def test_shipped_configs_parse():
    laptop = load_config(EVAL / "configs" / "laptop.toml")
    assert laptop.cer is None
    assert isinstance(laptop.per.model, RandomInit)
    assert isinstance(laptop.lid.model, Local) and laptop.lid.model.path == EVAL / "models" / "lang-id-voxlingua107-ecapa"
    box = load_config(EVAL / "configs" / "box.toml")
    assert box.cer.model == Hub("small") and box.speaker.reference_utterances == 10
    assert box.per.pair_counts == EVAL / "work" / "pairs.tsv"


def test_absent_sections_disable_metrics(tmp_path):
    config = parse_config({"threads": 2}, tmp_path)
    assert (config.cer, config.per, config.lid, config.speaker, config.f0, config.quality) == (None,) * 6


@pytest.mark.parametrize("raw", [
    {},
    {"threads": 2, "typo": 1},
    {"threads": 2, "metrics": {"per": {"model": {"hub": "x"}, "device": "cpu", "extra": 1}}},
    {"threads": 2, "metrics": {"per": {"model": {"hub": "x", "path": "y"}, "device": "cpu"}}},
    {"threads": 2, "metrics": {"lid": {"model": {"random_init": "x"}, "savedir": "s"}}},
    {"threads": 2, "metrics": {"speaker": {"model": {"hub": "x"}, "savedir": "s", "reference_utterances": 0}}},
    {"threads": "2"},
])
def test_bad_configs_fail(tmp_path, raw):
    with pytest.raises(ConfigError):
        parse_config(raw, tmp_path)
