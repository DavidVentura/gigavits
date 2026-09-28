import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

CATALOG = Path("/home/david/AndroidStudioProjects/Translator/app/src/main/assets/index_v6.json")
BUCKET = Path("/home/david/AndroidStudioProjects/bucket/tts/1")
REFERENCE_CONFIG = BUCKET / "en/en_US/amy/medium/en_US-amy-medium.onnx.json"
REFERENCE_MODEL = BUCKET / "en/en_US/amy/medium/en_US-amy-medium.mnn"
HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
FLORES = WORK / "flores200_dataset"

# espeak voice -> FLORES-200 file stem
FLORES_CODE = {
    "ar": "arb_Arab", "bg": "bul_Cyrl", "ca": "cat_Latn", "cs": "ces_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "el": "ell_Grek", "en-gb-x-rp": "eng_Latn", "en-us": "eng_Latn",
    "es": "spa_Latn", "es-419": "spa_Latn", "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn",
    "fr": "fra_Latn", "hi": "hin_Deva", "hu": "hun_Latn", "id": "ind_Latn", "is": "isl_Latn",
    "it": "ita_Latn", "ka": "kat_Geor", "lv": "lvs_Latn", "ml": "mal_Mlym", "nl": "nld_Latn",
    "nb": "nob_Latn", "pl": "pol_Latn", "pt-br": "por_Latn", "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl",
    "sk": "slk_Latn", "sl": "slv_Latn", "sq": "als_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn",
    "sw": "swh_Latn", "te": "tel_Telu", "tr": "tur_Latn", "uk": "ukr_Cyrl", "ur": "urd_Arab",
    "vi": "vie_Latn", "cmn": "zho_Hans",
    "az": "azj_Latn", "bn": "ben_Beng", "gu": "guj_Gujr", "he": "heb_Hebr", "kn": "kan_Knda",
    "mr": "mar_Deva", "ms": "zsm_Latn", "ta": "tam_Taml", "th": "tha_Thai", "ug": "uig_Arab",
    "bs": "bos_Latn", "et": "est_Latn", "hr": "hrv_Latn", "lt": "lit_Latn", "be": "bel_Cyrl",
    "yue": "yue_Hant", "ko": "kor_Hang", "ja": "jpn_Jpan",
}

# app language of non-Piper TTS packs -> espeak voice (None: espeak-ng has no voice)
NON_PIPER_ESPEAK = {
    "az": "az", "bn": "bn", "gu": "gu", "he": "he", "kn": "kn", "mr": "mr", "ms": "ms", "ta": "ta",
    "th": "th", "tl": None, "ug": "ug", "bs": "bs", "et": "et", "hr": "hr", "lt": "lt",
    "gl": None, "be": "be", "zh_hant": "yue", "ko": "ko", "ja": "ja",
}
NON_PIPER_ENGINES = ("mms", "coqui_vits", "cotovia_vits", "glowtts_hifigan", "sherpa_vits", "mimic3", "kokoro_mnn", "kokoro")

SPACELESS_SCRIPTS = {"Hans", "Hant", "Jpan", "Thai"}


class Kind(Enum):
    PIPER = "piper"
    NON_PIPER = "non-piper"


@dataclass(frozen=True)
class PiperVoice:
    pack: str
    language: str
    quality: str
    espeak_voice: str
    phoneme_type: str
    phoneme_id_map: dict

    @property
    def espeak_phonemized(self) -> bool:
        return self.phoneme_type == "espeak"


@dataclass(frozen=True)
class Run:
    voice: str
    kind: Kind
    flores: str
    app_languages: tuple
    packs: tuple


def load_catalog_packs() -> dict:
    return json.loads(CATALOG.read_text())["packs"]


def piper_voices() -> list:
    voices = []
    for key, pack in sorted(load_catalog_packs().items()):
        if not key.startswith("tts-piper-"):
            continue
        [config_file] = [f for f in pack["files"] if f["name"].endswith(".onnx.json")]
        config = json.loads((BUCKET / config_file["installPath"].split("bin/piper/", 1)[1]).read_text())
        voices.append(PiperVoice(
            pack=key,
            language=pack["language"],
            quality=pack["quality"],
            espeak_voice=config["espeak"]["voice"],
            phoneme_type=config.get("phoneme_type", "espeak"),
            phoneme_id_map=config["phoneme_id_map"],
        ))
    return voices


def runs() -> list:
    by_voice = {}
    for v in piper_voices():
        if not v.espeak_phonemized:
            continue
        by_voice.setdefault(v.espeak_voice, []).append(v)
    result = [
        Run(voice, Kind.PIPER, FLORES_CODE[voice],
            tuple(sorted({v.language for v in vs})), tuple(v.pack for v in vs))
        for voice, vs in sorted(by_voice.items())
    ]
    non_piper = {}
    for key, pack in sorted(load_catalog_packs().items()):
        if pack.get("feature") != "tts" or pack.get("engine") not in NON_PIPER_ENGINES or not pack.get("language"):
            continue
        voice = NON_PIPER_ESPEAK[pack["language"]]
        if voice is None or voice in by_voice:
            continue
        non_piper.setdefault(voice, []).append((pack["language"], key))
    result += [
        Run(voice, Kind.NON_PIPER, FLORES_CODE[voice],
            tuple(sorted({lang for lang, _ in items})), tuple(k for _, k in items))
        for voice, items in sorted(non_piper.items())
    ]
    return result


def non_piper_without_espeak() -> list:
    return sorted(
        (pack["language"], key)
        for key, pack in load_catalog_packs().items()
        if pack.get("feature") == "tts" and pack.get("engine") in NON_PIPER_ENGINES
        and pack.get("language") and NON_PIPER_ESPEAK[pack["language"]] is None
    )
