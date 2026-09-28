use piper_rs::{CoquiVitsModel, GlowTtsHifiganModel, KokoroMnnModel, MmsModel, PiperModel};
use std::path::Path;
use std::time::Instant;

const SENTENCES: &[&str] = &[
    "The old lighthouse keeper climbed the stairs every evening before sunset.",
    "Nobody expected the small bakery on the corner to become famous overnight.",
    "She folded the letter carefully and placed it inside the wooden drawer.",
    "Heavy rain delayed the train, so we waited in the station for an hour.",
    "Could you tell me where the nearest pharmacy is, please?",
    "The committee will announce its decision after reviewing all the proposals.",
    "A gentle breeze carried the smell of pine needles across the valley.",
    "He forgot his umbrella again, which was unfortunate given the forecast.",
    "Children were playing football in the park while their parents chatted nearby.",
    "The museum reopened with a new exhibition about ancient navigation techniques.",
    "Please remember to switch off the lights when you leave the office.",
    "Their journey through the mountains took longer than anyone had planned.",
    "I would rather read a good book than watch another television series.",
    "The orchestra rehearsed the final movement until it sounded perfect.",
    "Fresh vegetables from the market taste noticeably better than frozen ones.",
    "After the storm, the villagers gathered to repair the damaged bridge.",
];

enum Engine {
    Piper(PiperModel),
    Kokoro(KokoroMnnModel),
    Mms(MmsModel),
    Coqui(CoquiVitsModel),
    GlowTts(GlowTtsHifiganModel),
}

impl Engine {
    fn synthesize(&mut self, text: &str) -> (Vec<f32>, u32) {
        match self {
            Engine::Piper(m) => m.synthesize(text, None).unwrap(),
            Engine::Kokoro(m) => m.synthesize(text, Some(0), None).unwrap(),
            Engine::Mms(m) => m.synthesize(text, None, None).unwrap(),
            Engine::Coqui(m) => m.synthesize(text, None, None).unwrap(),
            Engine::GlowTts(m) => m.synthesize(text, None, None).unwrap(),
        }
    }
}

fn main() {
    let engine_name = std::env::args().nth(1).expect("engine: piper|kokoro");
    let model_path = std::env::args().nth(2).expect("piper config path or kokoro.mnn path");
    let target_audio_s: f64 = std::env::args()
        .nth(3)
        .expect("target audio seconds")
        .parse()
        .expect("target audio seconds must be a number");
    let espeak_data = std::env::var("ESPEAK_DATA").expect("ESPEAK_DATA");
    piper_rs::init_espeak(Path::new(&espeak_data)).expect("espeak init");

    let mut model = match engine_name.as_str() {
        "piper" => {
            let mnn = model_path.replace(".onnx.json", ".mnn");
            Engine::Piper(
                PiperModel::new(Path::new(&mnn), Path::new(&model_path), &piper_rs::Backend::Cpu)
                    .unwrap(),
            )
        }
        "kokoro" => {
            let voices = std::env::args().nth(4).expect("kokoro voices.bin path");
            Engine::Kokoro(
                KokoroMnnModel::new(Path::new(&model_path), Path::new(&voices), "en-us").unwrap(),
            )
        }
        "mimic3" => {
            let mnn = model_path.replace(".onnx.json", ".mnn");
            Engine::Piper(
                PiperModel::from_mimic3(Path::new(&mnn), Path::new(&model_path), &piper_rs::Backend::Cpu)
                    .unwrap(),
            )
        }
        "mms" => {
            let tokens = std::env::args().nth(4).expect("tokens.txt path");
            let lang = std::env::args().nth(5).expect("language code");
            Engine::Mms(
                MmsModel::new(Path::new(&model_path), Path::new(&tokens), &lang, &piper_rs::Backend::Cpu)
                    .unwrap(),
            )
        }
        "coqui" => {
            let config = std::env::args().nth(4).expect("config.json path");
            let lang = std::env::args().nth(5).expect("language code");
            Engine::Coqui(
                CoquiVitsModel::new(Path::new(&model_path), Path::new(&config), &lang, &piper_rs::Backend::Cpu)
                    .unwrap(),
            )
        }
        "glowtts" => {
            let vocoder = std::env::args().nth(4).expect("hifigan.mnn path");
            let lexicon = std::env::args().nth(5).expect("lexicon path");
            Engine::GlowTts(
                GlowTtsHifiganModel::new(
                    Path::new(&model_path),
                    Path::new(&vocoder),
                    Path::new(&lexicon),
                    &piper_rs::Backend::Cpu,
                )
                .unwrap(),
            )
        }
        other => panic!("unknown engine `{other}`"),
    };

    let own_language: Vec<String> = match (std::env::var("SENTENCES_TSV"), std::env::var("RUN")) {
        (Ok(tsv), Ok(run)) => std::fs::read_to_string(tsv)
            .unwrap()
            .lines()
            .filter_map(|l| {
                let mut cols = l.splitn(3, '\t');
                (cols.next()? == run).then(|| cols.nth(1).map(str::to_string))?
            })
            .take(40)
            .collect(),
        _ => SENTENCES.iter().map(|s| s.to_string()).collect(),
    };
    assert!(!own_language.is_empty(), "no sentences for RUN");

    let start = Instant::now();
    let (audio_s, utterances) = own_language
        .iter()
        .cycle()
        .scan(0.0f64, |acc, text| {
            if *acc >= target_audio_s {
                return None;
            }
            let (samples, sr) = model.synthesize(text);
            *acc += samples.len() as f64 / sr as f64;
            Some(*acc)
        })
        .fold((0.0, 0usize), |(_, n), acc| (acc, n + 1));
    let wall = start.elapsed().as_secs_f64();
    println!(
        "utterances={utterances} audio_s={audio_s:.1} wall_s={wall:.2} rtf={:.4} x_realtime={:.1}",
        wall / audio_s,
        audio_s / wall
    );
}
