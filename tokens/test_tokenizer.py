import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from remap import LayoutError, remap, remap_ids, teacher_input
from tokenizer import NotNfd, Table, UnknownSymbol, UnsupportedLanguage, piper_input, symbol_ids, tokenize

HERE = Path(__file__).resolve().parent
CORPUS_PATH = HERE.parent / "coverage" / "work" / "phonemized.jsonl"
UNSUPPORTED = {"cmn", "yue", "th"}

RETROFLEX = "̢"
PHARYNGEALIZED = "ˤ"
GLOTTAL = "ʔ"
VOICELESS = "̊"
PALATALIZED = "ʲ"
NON_SYLLABIC = "̯"
CENTRALIZED = "̈"
EXTRA_SHORT = "̆"
FORTIS = "͈"
SYLLABIC = "̩"

TABLE: Table
KEY_BY_ID: dict[int, str]
CORPUS: dict[tuple[str, int], str]  # as piper-rs feeds it (trimmed)
RAW: dict[tuple[str, int], str]  # as espeak_phonemize returns it


def setUpModule():
    global TABLE, KEY_BY_ID, CORPUS, RAW
    TABLE = Table.load(HERE / "table.json")
    KEY_BY_ID = {v: k for k, v in TABLE.ids.items()}
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"{CORPUS_PATH} is missing; run coverage/run.sh")
    CORPUS, RAW = {}, {}
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            RAW[(r["run"], r["idx"])] = r["phonemes"]
            CORPUS[(r["run"], r["idx"])] = piper_input(r["phonemes"])


def replaced(excerpt: str, replacements: dict[int, str]) -> list[str]:
    keys = list(excerpt)
    for i, key in replacements.items():
        keys[i] = key
    return keys


def prefix_map(n: int) -> dict[str, list[int]]:
    """An old Piper voice map: every espeak voice map in the bucket is the shared table's ids 0..n-1."""
    return {k: [i] for k, i in TABLE.ids.items() if i < n}


class SpecialCases(unittest.TestCase):
    def assertExcerpt(self, run: str, idx: int, excerpt: str, replacements: dict[int, str]):
        phonemes = CORPUS[(run, idx)]
        start = phonemes.index(excerpt)
        ids = symbol_ids(phonemes, TABLE.language(run), TABLE)
        got = [KEY_BY_ID[x] for x in ids[start:start + len(excerpt)]]
        self.assertEqual(got, replaced(excerpt, replacements), f"{run}:{idx} {excerpt!r}")

    def test_sequence_layout(self):
        phonemes = CORPUS[("de", 0)]
        ids = tokenize(phonemes, TABLE.language("de"), TABLE)
        self.assertEqual(len(ids), 2 * len(phonemes) + 3)
        self.assertEqual(ids[:2], [TABLE.bos, TABLE.pad])
        self.assertEqual(ids[-1], TABLE.eos)
        self.assertTrue(all(x == TABLE.pad for x in ids[1:-1:2]))
        self.assertNotIn(TABLE.bos, ids[2:-1:2])

    def test_hindi_retroflex_r(self):
        self.assertExcerpt("hi", 3, "lˈʊr.hək", {4: RETROFLEX})
        self.assertExcerpt("hi", 27, "cʰˈeːr. sˈʌktˌe", {6: RETROFLEX})
        self.assertExcerpt("hi", 20, "eːɡi.  bˈʌcːo\u0303", {})

    def test_other_retroflex_r_languages(self):
        self.assertExcerpt("ml", 0, "ɡəɭˌaːr.ccɐ, s", {7: RETROFLEX})
        self.assertExcerpt("bn", 3, "ɡˈar. kˈaɟeɾ", {4: RETROFLEX})
        self.assertExcerpt("gu", 91, "aːɡˈʌr. kˈʌhju\u0303", {6: RETROFLEX})
        self.assertExcerpt("ur", 3, "ɡˈaːr.i kˈaː", {5: RETROFLEX})
        self.assertExcerpt("ur", 60, "pˈʌkər. hɛɟˈeːs", {6: RETROFLEX})

    def test_mid_chunk_clause_period(self):
        self.assertExcerpt("bn", 21, "hˈɔbe. pˈɔr.aɾ", {11: RETROFLEX})
        self.assertExcerpt("mr", 259, "ːjˈaːdi. aːvˈʌʃj", {})
        self.assertExcerpt("ko", 2, "wˈɐt-t-ɐ. ɡˈomˈ", {4: FORTIS, 6: FORTIS})

    def test_arabic_emphatic_vowels(self):
        self.assertExcerpt("ar", 0, "t̪ˈa.ːbʕaːt", {4: PHARYNGEALIZED})
        self.assertExcerpt("ar", 8, "mˈarra. ʕˈalaː", {6: PHARYNGEALIZED})

    def test_syllabic_marker(self):
        self.assertExcerpt("ar", 63, "was̪s̪-ˈiːn", {6: SYLLABIC})
        self.assertExcerpt("da", 11, "dˈaj-əns", {4: SYLLABIC})

    def test_danish_stod(self):
        self.assertExcerpt("da", 0, "fʁˈ?amstʔel", {3: GLOTTAL})
        self.assertExcerpt("da", 243, "m?ɑj?  ʋes", {1: GLOTTAL})

    def test_icelandic_voiceless_sonorants(self):
        self.assertExcerpt("is", 0, "lˈaɪɡhn#adeɪld", {7: VOICELESS})
        self.assertExcerpt("is", 0, "dlidɪtl# pərˈɛn", {7: VOICELESS})

    def test_estonian_palatalization(self):
        self.assertExcerpt("et", 2, "mˈeːd^ia", {5: PALATALIZED})
        self.assertExcerpt("et", 5, "ˌit^siˈoːn", {3: PALATALIZED})

    def test_russian_centralized_u_and_silent_i(self):
        self.assertExcerpt("ru", 12, 'tʲu"rɪ^mˈɑ', {3: CENTRALIZED, 6: NON_SYLLABIC})

    def test_latvian_short_variants(self):
        self.assertExcerpt("lv", 22, "ʋˈa`ja`ɡ", {3: EXTRA_SHORT, 6: EXTRA_SHORT})
        self.assertExcerpt("lv", 5, "ˈuz-ʋareːt", {3: EXTRA_SHORT})

    def test_korean_fortis(self):
        self.assertExcerpt("ko", 0, "ndweˌʌt-t-ɐ.", {7: FORTIS, 9: FORTIS})
        self.assertExcerpt("ko", 2, "qiqˌɐ q-ˈɯnnɐn", {7: FORTIS})

    def test_french_clitic_vowels(self):
        self.assertExcerpt("fr", 0, "ifˈik də- lekˈɔl", {8: EXTRA_SHORT})
        self.assertExcerpt("fr", 8, "le-z elɛks", {2: EXTRA_SHORT})

    def test_vietnamese_tones(self):
        self.assertExcerpt("vi", 5, "zˈe-2ɲ", {3: EXTRA_SHORT, 4: "vi:tone2"})
        self.assertExcerpt("vi", 6, "xˈe-ɜc", {3: EXTRA_SHORT, 4: "vi:tone3"})
        self.assertExcerpt("vi", 20, "kˈə1w", {3: "vi:tone1"})
        self.assertExcerpt("vi", 19, "ðˈe7", {3: "vi:tone7"})

    def test_vietnamese_english_loan_keeps_vowel(self):
        self.assertExcerpt("vi", 54, "ˈɜːkʌtsk", {})

    def test_japanese_diacritics(self):
        self.assertExcerpt("ja", 0, "sˌo̞ɯᵝdʑɯᵝˈɯ", {})

    def test_persian_default_tone_marker(self):
        self.assertExcerpt("fa", 0, "q1ˈɑbel", {})


class Rejections(unittest.TestCase):
    def assertUnknown(self, phonemes: str, language: str, char: str):
        with self.assertRaises(UnknownSymbol) as cm:
            symbol_ids(phonemes, TABLE.language(language), TABLE)
        self.assertEqual(cm.exception.char, char)

    def test_phonetic_characters_outside_their_language(self):
        self.assertUnknown("ˈaɪn#a", "de", "#")
        self.assertUnknown("mˈeːd^ia", "de", "^")
        self.assertUnknown("zˈe2ɲ", "de", "2")
        self.assertUnknown("t̪ˈa.ːb", "de", ".")
        self.assertUnknown("fʁˈ?am", "sv", "?")
        self.assertUnknown("ʋˈa`ja", "lt", "`")

    def test_phonetic_characters_outside_their_context(self):
        self.assertUnknown("fʁˈ?em", "da", "?")
        self.assertUnknown("ˈa#a", "is", "#")
        self.assertUnknown("ˈa-a", "de", "-")
        self.assertUnknown("ˈb2a", "vi", "2")

    def test_clause_punctuation_needs_the_following_space_or_the_end(self):
        self.assertUnknown("ˈa,b", "de", ",")
        self.assertUnknown("ˈa.b", "de", ".")
        self.assertEqual(symbol_ids("ˈa.", TABLE.language("de"), TABLE)[-1], TABLE.ids["."])

    def test_unknown_character(self):
        self.assertUnknown("ˈa§b", "de", "§")

    def test_bos_eos_pad_characters(self):
        self.assertUnknown("ˈa^b", "de", "^")
        self.assertUnknown("ˈa_b", "de", "_")
        self.assertUnknown("ˈa$b", "de", "$")

    def test_not_nfd(self):
        with self.assertRaises(NotNfd):
            symbol_ids("\u00e7a", TABLE.language("de"), TABLE)

    def test_unsupported_languages(self):
        for code in [*UNSUPPORTED, "xx"]:
            with self.assertRaises(UnsupportedLanguage):
                TABLE.language(code)


class FullCorpus(unittest.TestCase):
    def test_every_supported_sentence_tokenizes(self):
        runs = {run for run, _ in CORPUS}
        self.assertEqual(runs & UNSUPPORTED, UNSUPPORTED)
        failures = defaultdict(Counter)
        checked = Counter()
        punctuation_ids = {TABLE.ids[c] for c in TABLE.clause_punctuation}
        legacy_ids = {i for k, i in TABLE.ids.items() if TABLE.classes[k] in ("legacy", "special")}
        for (run, idx), phonemes in sorted(CORPUS.items()):
            if run in UNSUPPORTED:
                continue
            try:
                ids = symbol_ids(phonemes, TABLE.language(run), TABLE)
            except UnknownSymbol as e:
                failures[run][e.char] += 1
                continue
            checked[run] += 1
            self.assertEqual(len(ids), len(phonemes))
            # trimming only drops the edge spaces: every other position keeps its token
            raw = RAW[(run, idx)]
            lead = len(raw) - len(raw.lstrip())
            self.assertEqual(ids, symbol_ids(raw, TABLE.language(run), TABLE)[lead:lead + len(phonemes)], (run, idx))
            counts = Counter(ids)
            self.assertEqual(counts[TABLE.ids[" "]], phonemes.count(" "), (run, idx))
            for c in ",;:!":
                self.assertEqual(counts[TABLE.ids[c]], phonemes.count(c), (run, idx, c))
            n_punct = sum(counts[i] for i in punctuation_ids)
            self.assertLessEqual(n_punct, sum(phonemes.count(c) for c in TABLE.clause_punctuation))
            unexpected = {KEY_BY_ID[i] for i in counts if i in legacy_ids} - ({"1"} if run == "fa" else set())
            self.assertEqual(unexpected, set(), (run, idx))
        self.assertEqual({r: dict(c) for r, c in failures.items()}, {})
        self.assertEqual(len(checked), len(runs - UNSUPPORTED))
        self.assertTrue(all(n == 400 for n in checked.values()), checked)


class Remap(unittest.TestCase):
    def test_lossless_voice_aligns_position_by_position(self):
        phonemes = CORPUS[("is", 0)]
        result = remap(phonemes, TABLE.language("is"), prefix_map(162), TABLE)
        self.assertTrue(result.aligned)
        differing = {
            (KEY_BY_ID[t], KEY_BY_ID[s])
            for t, s in zip(result.teacher.ids, result.student_ids) if t != s
        }
        self.assertEqual(differing, {("#", VOICELESS)})

    def test_icelandic_130_symbol_voice_is_lossy(self):
        result = remap(CORPUS[("is", 0)], TABLE.language("is"), prefix_map(130), TABLE)
        self.assertFalse(result.aligned)
        self.assertEqual({d.char for d in result.teacher.dropped}, {"#"})

    def test_vietnamese_tone_digits(self):
        phonemes = CORPUS[("vi", 5)]
        kept = remap(phonemes, TABLE.language("vi"), prefix_map(154), TABLE)
        self.assertTrue(kept.aligned)
        self.assertIn((TABLE.ids["2"], TABLE.ids["vi:tone2"]), set(zip(kept.teacher.ids, kept.student_ids)))
        lossy = remap(phonemes, TABLE.language("vi"), prefix_map(130), TABLE)
        self.assertFalse(lossy.aligned)
        self.assertIn("2", {d.char for d in lossy.teacher.dropped})

    def test_remap_ids_matches_tokenize_for_lossless_input(self):
        for run, idx in [("hi", 3), ("da", 243), ("ar", 8), ("vi", 5), ("fr", 0)]:
            language = TABLE.language(run)
            teacher = teacher_input(CORPUS[(run, idx)], prefix_map(161))
            self.assertEqual(teacher.dropped, ())
            self.assertEqual(
                remap_ids(list(teacher.ids), language, prefix_map(161), TABLE),
                tokenize(CORPUS[(run, idx)], language, TABLE),
            )

    def test_remap_ids_rejects_bad_layout(self):
        with self.assertRaises(LayoutError):
            remap_ids([1, 0, 14, 2], TABLE.language("de"), prefix_map(162), TABLE)

    def test_piper_whitespace_split_quirk_is_not_aligned(self):
        teacher = teacher_input("a b", prefix_map(162))
        self.assertTrue(teacher.whitespace_split)
        self.assertEqual(teacher.ids, (1, 0, 14, 0, 15, 0, 2))


if __name__ == "__main__":
    unittest.main()
