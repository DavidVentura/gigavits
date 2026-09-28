import pytest

from config import load_config
from phonemize import phonemize
from repo import EVAL

CONFIG = load_config(EVAL / "configs" / "laptop.toml").phonemizer


@pytest.mark.skipif(not (CONFIG.binary.exists() and CONFIG.reference_model.exists()), reason="piper-rs phonemizer not built here")
def test_order_is_kept_across_languages():
    out = phonemize(CONFIG, [("it", "Buongiorno a tutti."), ("de", "Das ist ein Test."), ("it", "Grazie.")])
    assert out[1] == "das ɪst aɪn tˈɛst. "
    assert out[0].startswith("bʊondʒˈɔrno") and out[2].startswith("ɡrˈatsje")
