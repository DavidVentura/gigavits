//! Teacher runtime for the data generator.
//!
//! Phonemizes text exactly as the app does (piper-rs + espeak-ng) and renders the teachers that
//! only exist as MNN models (Kokoro, MMS, Coqui, mimic3, GlowTTS). Piper teachers are rendered by
//! the Python driver with onnxruntime, because MNN cannot return the `durations` output and cannot
//! be seeded; for them this process only phonemizes and reports the IDs piper-rs would feed.
//!
//! `teacher-rt convert <in.onnx> <out.mnn>` converts a teacher to MNN without weight quantization:
//! the app ships int8 weights, and teachers must render from fp32 (PLAN Phase 2).
//!
//! One process serves one teacher: `teacher-rt <espeak-data> <spec json> <response fd>`. The driver
//! writes one JSON request per line on stdin and reads one JSON response per line from the response
//! fd, a pipe it passes in, because MNN prints its CPU banner to stdout. Any failure panics; the
//! driver treats a dead process as a failed job.

use std::io::{BufRead, BufWriter, Write};
use std::os::fd::FromRawFd;
use std::path::{Path, PathBuf};

use piper_rs::{CoquiVitsModel, GlowTtsHifiganModel, KokoroMnnModel, MmsModel, PiperModel};
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

#[derive(Deserialize)]
#[serde(tag = "engine", rename_all = "snake_case", deny_unknown_fields)]
enum Spec {
    Espeak { espeak: String },
    Piper { config: PathBuf, mnn: PathBuf },
    Kokoro { model: PathBuf, voices: PathBuf, voice: String, espeak: String },
    KokoroJa { model: PathBuf, voices: PathBuf, voice: String, dict: PathBuf, espeak: String },
    Mms { model: PathBuf, tokens: PathBuf, espeak: String },
    Coqui { model: PathBuf, config: PathBuf, espeak: String },
    Mimic3 { config: PathBuf, mnn: PathBuf, espeak: String },
    Glowtts { model: PathBuf, vocoder: PathBuf, lexicon: PathBuf, espeak: String },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    text: String,
    render: Option<Render>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Render {
    speed: f32,
    audio_out: PathBuf,
}

#[derive(Serialize)]
struct Response {
    /// The espeak string exactly as piper-rs `espeak_phonemize` returns it (untrimmed, NFD).
    phonemes: String,
    teacher: Option<TeacherInput>,
    audio: Option<AudioOut>,
}

/// What piper-rs feeds a Piper voice for this text.
#[derive(Serialize)]
struct TeacherInput {
    /// The string mapped to IDs: trimmed, NFD; the espeak string, or the text for `text` voices.
    fed: String,
    /// Its symbol IDs, without BOS/PAD/EOS.
    piper_ids: Vec<i64>,
}

#[derive(Serialize)]
struct AudioOut {
    sample_rate: u32,
    samples: usize,
}

enum KokoroFrontend {
    // piper-rs phonemizes Kokoro with the same espeak call as Piper; checked per request.
    Espeak,
    // Japanese Kokoro reads MeCab output, so its input differs from the student's espeak string.
    MeCab,
}

enum Renderer {
    Kokoro { model: KokoroMnnModel, sid: i64, frontend: KokoroFrontend },
    Mms(MmsModel),
    Coqui(CoquiVitsModel),
    Mimic3(PiperModel),
    GlowTts(GlowTtsHifiganModel),
}

enum Teacher {
    Phonemizer { espeak: String },
    // text_input: the voice reads characters (`phoneme_type: text`), not espeak phonemes
    Piper { model: PiperModel, espeak: String, text_input: bool },
    Renderer { renderer: Renderer, espeak: String },
}

/// Mirror of piper-rs `espeak_phonemize` (private there), which the app uses for every espeak
/// voice. Piper and Kokoro requests assert equality with piper-rs's own output, so a drift in
/// piper-rs fails loudly instead of silently changing the student input for the other engines.
fn espeak_phonemes(text: &str, espeak: &str) -> String {
    let voice = match espeak {
        "zh" => "cmn",
        other => other,
    };
    espeak_rs::text_to_phonemes(text, voice, None)
        .unwrap_or_else(|e| panic!("espeak `{voice}` failed on {text:?}: {e}"))
        .join(" ")
        .nfd()
        .collect()
}

#[derive(Deserialize)]
struct PiperConfigEspeak {
    voice: String,
}

#[derive(Deserialize)]
struct PiperConfigHead {
    espeak: PiperConfigEspeak,
    #[serde(default = "espeak_type")]
    phoneme_type: String,
}

fn espeak_type() -> String {
    "espeak".to_string()
}

fn load(spec: Spec) -> Teacher {
    let cpu = &piper_rs::Backend::Cpu;
    match spec {
        Spec::Espeak { espeak } => Teacher::Phonemizer { espeak },
        Spec::Piper { config, mnn } => {
            let head: PiperConfigHead =
                serde_json::from_reader(std::fs::File::open(&config).expect("piper config")).expect("piper config json");
            let model = PiperModel::new(&mnn, &config, cpu).unwrap_or_else(|e| panic!("load {}: {e}", mnn.display()));
            let text_input = match head.phoneme_type.as_str() {
                "espeak" => false,
                "text" => true,
                other => panic!("unsupported phoneme_type `{other}`"),
            };
            Teacher::Piper { model, espeak: head.espeak.voice, text_input }
        }
        Spec::Kokoro { model, voices, voice, espeak } => {
            let m = KokoroMnnModel::new(&model, &voices, &espeak).unwrap_or_else(|e| panic!("load kokoro: {e}"));
            let sid = kokoro_sid(&m, &voice);
            Teacher::Renderer { renderer: Renderer::Kokoro { model: m, sid, frontend: KokoroFrontend::Espeak }, espeak }
        }
        Spec::KokoroJa { model, voices, voice, dict, espeak } => {
            let mut m = KokoroMnnModel::new(&model, &voices, "ja").unwrap_or_else(|e| panic!("load kokoro: {e}"));
            m.load_japanese_dict(&dict.to_string_lossy()).unwrap_or_else(|e| panic!("load mecab dict: {e}"));
            let sid = kokoro_sid(&m, &voice);
            Teacher::Renderer { renderer: Renderer::Kokoro { model: m, sid, frontend: KokoroFrontend::MeCab }, espeak }
        }
        Spec::Mms { model, tokens, espeak } => {
            let m = MmsModel::new(&model, &tokens, &espeak, cpu).unwrap_or_else(|e| panic!("load mms: {e}"));
            Teacher::Renderer { renderer: Renderer::Mms(m), espeak }
        }
        Spec::Coqui { model, config, espeak } => {
            let m = CoquiVitsModel::new(&model, &config, &espeak, cpu).unwrap_or_else(|e| panic!("load coqui: {e}"));
            Teacher::Renderer { renderer: Renderer::Coqui(m), espeak }
        }
        Spec::Mimic3 { config, mnn, espeak } => {
            let m = PiperModel::from_mimic3(&mnn, &config, cpu).unwrap_or_else(|e| panic!("load mimic3: {e}"));
            Teacher::Renderer { renderer: Renderer::Mimic3(m), espeak }
        }
        Spec::Glowtts { model, vocoder, lexicon, espeak } => {
            let m = GlowTtsHifiganModel::new(&model, &vocoder, &lexicon, cpu).unwrap_or_else(|e| panic!("load glowtts: {e}"));
            Teacher::Renderer { renderer: Renderer::GlowTts(m), espeak }
        }
    }
}

fn kokoro_sid(model: &KokoroMnnModel, voice: &str) -> i64 {
    *model
        .voices()
        .and_then(|v| v.get(voice))
        .unwrap_or_else(|| panic!("kokoro voice `{voice}` not in voices file"))
}

impl Renderer {
    fn render(&mut self, text: &str, speed: f32) -> (Vec<f32>, u32) {
        let result = match self {
            Renderer::Kokoro { model, sid, .. } => model.synthesize(text, Some(*sid), Some(speed)),
            Renderer::Mms(m) => m.synthesize(text, None, Some(speed)),
            Renderer::Coqui(m) => m.synthesize(text, None, Some(speed)),
            Renderer::Mimic3(m) => m.synthesize_with_options(text, None, Some(1.0 / speed)),
            Renderer::GlowTts(m) => m.synthesize(text, None, Some(speed)),
        };
        result.unwrap_or_else(|e| panic!("render {text:?}: {e}"))
    }

    fn check_phonemes(&mut self, text: &str, phonemes: &str) {
        if let Renderer::Kokoro { model, frontend: KokoroFrontend::Espeak, .. } = self {
            let own = model.phonemize(text).unwrap_or_else(|e| panic!("kokoro phonemize {text:?}: {e}"));
            assert_eq!(own, phonemes, "kokoro phonemes differ from the espeak mirror for {text:?}");
        }
    }
}

fn handle(teacher: &mut Teacher, request: Request) -> Response {
    match teacher {
        Teacher::Phonemizer { espeak } => {
            assert!(request.render.is_none(), "phonemizer cannot render");
            Response { phonemes: espeak_phonemes(&request.text, espeak), teacher: None, audio: None }
        }
        Teacher::Piper { model, espeak, text_input } => {
            assert!(request.render.is_none(), "piper teachers are rendered with onnxruntime by the driver");
            let own = model.phonemize(&request.text).unwrap_or_else(|e| panic!("phonemize {:?}: {e}", request.text));
            let phonemes = espeak_phonemes(&request.text, espeak);
            if !*text_input {
                assert_eq!(own, phonemes, "piper-rs phonemes differ from the espeak mirror");
            }
            // PiperModel::synthesize_phonemes trims and re-normalizes before mapping to IDs.
            let fed: String = own.trim().nfd().collect();
            let piper_ids = model.debug_phoneme_string(&fed).expect("piper frontend").token_ids;
            Response { phonemes, teacher: Some(TeacherInput { fed, piper_ids }), audio: None }
        }
        Teacher::Renderer { renderer, espeak } => {
            let render = request.render.expect("render engines need a render request");
            let phonemes = espeak_phonemes(&request.text, espeak);
            renderer.check_phonemes(&request.text, &phonemes);
            let (samples, sample_rate) = renderer.render(&request.text, render.speed);
            write_f32(&render.audio_out, &samples);
            Response { phonemes, teacher: None, audio: Some(AudioOut { sample_rate, samples: samples.len() }) }
        }
    }
}

fn write_f32(path: &Path, samples: &[f32]) {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn convert(onnx: &Path, mnn: &Path) {
    mnn_sys::convert_onnx_to_mnn(onnx, mnn, mnn_sys::WeightQuant::None)
        .unwrap_or_else(|e| panic!("convert {}: {e}", onnx.display()));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [command, onnx, mnn] = args.as_slice() {
        if command == "convert" {
            convert(Path::new(onnx), Path::new(mnn));
            return;
        }
    }
    let mut args = args.into_iter();
    let usage = "usage: teacher-rt <espeak-data> <spec json> <response fd>";
    let espeak_data = PathBuf::from(args.next().expect(usage));
    let spec: Spec = serde_json::from_str(&args.next().expect(usage)).expect("spec json");
    let response_fd: i32 = args.next().expect(usage).parse().expect("response fd must be an integer");
    // SAFETY: the driver opens this fd for us and nothing else in the process uses it.
    let responses = unsafe { std::fs::File::from_raw_fd(response_fd) };
    piper_rs::init_espeak(&espeak_data).expect("espeak init");
    let mut teacher = load(spec);

    let stdin = std::io::stdin();
    let mut out = BufWriter::new(responses);
    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        let request: Request = serde_json::from_str(&line).unwrap_or_else(|e| panic!("request {line:?}: {e}"));
        let response = handle(&mut teacher, request);
        serde_json::to_writer(&mut out, &response).unwrap();
        out.write_all(b"\n").unwrap();
        out.flush().unwrap();
    }
}
