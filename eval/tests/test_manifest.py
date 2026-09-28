import pytest

from manifest import ManifestError, ReferenceVoice, VoiceKey, parse_manifest, write_manifest
from repo import ROOT

SAMPLES = ROOT / "voices" / "samples" / "manifest.tsv"


def test_samples_manifest_parses():
    utterances = parse_manifest(SAMPLES)
    assert len(utterances) == 450
    first = utterances[0]
    assert first.key == VoiceKey("ar_JO-kareem-low", "0")
    assert first.path == SAMPLES.parent / "ar_JO-kareem-low__0__0.wav"
    assert first.sample_rate == 16000
    assert first.espeak_phonemes is None and first.reference_voice is None
    # Quotes in the text survive (the file is not CSV-quoted).
    assert any(u.text.count('"') for u in utterances)


def header(extra=""):
    return "voice\tspeaker\tespeak\tidx\tfile\tsample_rate\ttext" + extra + "\n"


def test_optional_columns(tmp_path):
    path = tmp_path / "m.tsv"
    path.write_text(header("\treference_voice\treference_wav\tcpu_s\tespeak_phonemes")
                    + "v\t0\tde\t0\ta.wav\t22050\tDas.\tseed_1/v|0\tref/a.wav\t0.25\tdas. \n", encoding="utf-8")
    u = parse_manifest(path)[0]
    assert u.reference_voice == ReferenceVoice("seed_1", VoiceKey("v", "0"))
    assert u.reference_wav == tmp_path / "ref" / "a.wav"
    assert u.cpu_s == 0.25
    assert u.espeak_phonemes == "das. "


@pytest.mark.parametrize("body", [
    "v\t0\tde\tx\ta.wav\t22050\tDas.\n",
    "v\t0\tde\t0\t\t22050\tDas.\n",
])
def test_bad_rows_fail_with_line_number(tmp_path, body):
    path = tmp_path / "m.tsv"
    path.write_text(header() + body, encoding="utf-8")
    with pytest.raises(ManifestError, match="m.tsv:2"):
        parse_manifest(path)


def test_missing_column_fails(tmp_path):
    path = tmp_path / "m.tsv"
    path.write_text("voice\tspeaker\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="no column"):
        parse_manifest(path)


def test_duplicates_fail(tmp_path):
    path = tmp_path / "m.tsv"
    path.write_text(header() + "v\t0\tde\t0\ta.wav\t22050\tA.\n" + "v\t0\tde\t0\tb.wav\t22050\tB.\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="duplicate"):
        parse_manifest(path)


@pytest.mark.parametrize("text", ["v", "v|", "|0", "v|0|1"])
def test_bad_voice_keys(text):
    with pytest.raises(ManifestError):
        VoiceKey.parse(text)


def test_reference_voice_needs_a_set():
    with pytest.raises(ManifestError):
        ReferenceVoice.parse("v|0")
    assert str(ReferenceVoice.parse("orig/v|0")) == "orig/v|0"


def test_write_then_parse_roundtrip(tmp_path):
    rows = [{"voice": "v", "speaker": "0", "espeak": "es", "idx": 0, "file": "a.wav", "sample_rate": 22050,
             "text": 'Dijo "hola".', "espeak_phonemes": "dˈixo ˈola. ", "cpu_s": "0.1"}]
    path = tmp_path / "m.tsv"
    write_manifest(path, rows)
    u = parse_manifest(path)[0]
    assert u.text == 'Dijo "hola".' and u.espeak_phonemes == "dˈixo ˈola. " and u.cpu_s == 0.1


def test_write_rejects_tabs(tmp_path):
    with pytest.raises(ManifestError):
        write_manifest(tmp_path / "m.tsv", [{"voice": "v", "speaker": "0", "espeak": "es", "idx": 0, "file": "a.wav",
                                             "sample_rate": 1, "text": "a\tb"}])
