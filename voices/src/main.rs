use piper_rs::{CoquiVitsModel, GlowTtsHifiganModel, KokoroMnnModel, MmsModel, PiperModel};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

const SENTENCES_PER_SPEAKER: usize = 3;
const MAX_SPEAKERS_PER_MODEL: usize = 4;

#[derive(Deserialize)]
struct ESpeak {
    voice: String,
}

#[derive(Deserialize)]
struct VoiceConfig {
    espeak: ESpeak,
    #[serde(default)]
    speaker_id_map: HashMap<String, i64>,
}

enum Source {
    Piper { config: PathBuf, mnn: PathBuf },
    Kokoro { voice: String, espeak: String },
    // Teachers from other engines; the samples only need audio, so their text is not phonemized here.
    Other { engine: OtherEngine, espeak: String },
}

enum OtherEngine {
    Mms { model: PathBuf, tokens: PathBuf },
    Coqui { model: PathBuf, config: PathBuf },
    GlowTts { model: PathBuf, vocoder: PathBuf, lexicon: PathBuf },
    Mimic3 { config: PathBuf },
    KokoroJa { voice: String, dict: PathBuf },
}

struct Voice {
    name: String,
    source: Source,
}

const KOKORO_MODEL: &str = "/home/david/AndroidStudioProjects/bucket/tts/1/kokoro_mnn/kokoro.mnn";
const KOKORO_VOICES: &str = "/home/david/AndroidStudioProjects/bucket/support/1/voices-v1.0.bin";

fn main() {
    let mut args = std::env::args().skip(1);
    let voice_list = args.next().expect("file listing <voice name>\\t<config path>");
    let sentences_tsv = args.next().expect("sentences.tsv (run, app_lang, text)");
    let out_dir = PathBuf::from(args.next().expect("output dir"));
    let espeak_data = std::env::var("ESPEAK_DATA").expect("ESPEAK_DATA");
    piper_rs::init_espeak(Path::new(&espeak_data)).expect("espeak init");
    std::fs::create_dir_all(&out_dir).unwrap();

    let sentences = load_sentences(&sentences_tsv);
    let voices: Vec<Voice> = std::fs::read_to_string(&voice_list)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let (name, spec) = line.split_once('\t').expect("voice list line");
            // Kokoro lines are `kokoro-<voice>\tkokoro:<espeak voice>`; other engines are
            // `<engine>:<language>:<path>[|<path>...]`.
            let source = match spec.split_once(':') {
                Some(("kokoro", espeak)) => Source::Kokoro {
                    voice: name.trim_start_matches("kokoro-").to_string(),
                    espeak: espeak.to_string(),
                },
                Some((engine, rest)) => {
                    let (espeak, paths) = rest.split_once(':').expect("engine line: <engine>:<lang>:<paths>");
                    let p: Vec<PathBuf> = paths.split('|').map(PathBuf::from).collect();
                    let engine = match engine {
                        "mms" => OtherEngine::Mms { model: p[0].clone(), tokens: p[1].clone() },
                        "coqui" => OtherEngine::Coqui { model: p[0].clone(), config: p[1].clone() },
                        "glowtts" => OtherEngine::GlowTts { model: p[0].clone(), vocoder: p[1].clone(), lexicon: p[2].clone() },
                        "mimic3" => OtherEngine::Mimic3 { config: p[0].clone() },
                        "kokoro_ja" => OtherEngine::KokoroJa {
                            voice: name.trim_start_matches("kokoro-").to_string(),
                            dict: p[0].clone(),
                        },
                        other => panic!("unknown engine `{other}`"),
                    };
                    Source::Other { engine, espeak: espeak.to_string() }
                }
                None => Source::Piper {
                    config: PathBuf::from(spec),
                    mnn: PathBuf::from(spec.replace(".onnx.json", ".mnn")),
                },
            };
            Voice { name: name.to_string(), source }
        })
        .collect();

    let mut manifest = std::fs::File::create(out_dir.join("manifest.tsv")).unwrap();
    writeln!(manifest, "voice\tspeaker\tespeak\tidx\tfile\tsample_rate\tphonemes\ttext").unwrap();
    for voice in &voices {
        let (config, mnn) = match &voice.source {
            Source::Kokoro { voice: kokoro_voice, espeak } => {
                render_kokoro(&voice.name, kokoro_voice, espeak, &sentences, &out_dir, &mut manifest);
                continue;
            }
            Source::Other { engine, espeak } => {
                render_other(&voice.name, engine, espeak, &sentences, &out_dir, &mut manifest);
                continue;
            }
            Source::Piper { config, mnn } => (config, mnn),
        };
        let config_path = config;
        let config: VoiceConfig =
            serde_json::from_reader(std::fs::File::open(config_path).unwrap()).unwrap();
        let espeak = config.espeak.voice.clone();
        let texts = sentences_for(&sentences, &espeak)
            .unwrap_or_else(|| panic!("no sentences for espeak voice `{espeak}` ({})", voice.name));
        let mut model = PiperModel::new(mnn, config_path, &piper_rs::Backend::Cpu)
            .unwrap_or_else(|e| panic!("load {}: {e}", voice.name));
        for (speaker, sid) in pick_speakers(&config.speaker_id_map) {
            for (idx, text) in texts.iter().take(SENTENCES_PER_SPEAKER).enumerate() {
                let (samples, sr) = model.synthesize(text, sid).unwrap();
                let phonemes = model
                    .debug_frontend(text)
                    .unwrap()
                    .tokens
                    .iter()
                    .filter(|t| t.chars().next().is_some_and(char::is_alphabetic))
                    .count();
                let file = format!("{}__{}__{idx}.wav", voice.name, speaker);
                write_wav(&out_dir.join(&file), &samples, sr);
                writeln!(
                    manifest,
                    "{}\t{speaker}\t{espeak}\t{idx}\t{file}\t{sr}\t{phonemes}\t{}",
                    voice.name,
                    text.replace('\t', " ")
                )
                .unwrap();
            }
        }
        eprintln!("rendered {}", voice.name);
    }
}

fn render_kokoro(
    name: &str,
    kokoro_voice: &str,
    espeak: &str,
    sentences: &HashMap<String, Vec<String>>,
    out_dir: &Path,
    manifest: &mut std::fs::File,
) {
    let texts = sentences_for(sentences, espeak)
        .unwrap_or_else(|| panic!("no sentences for espeak voice `{espeak}` ({name})"));
    let mut model = KokoroMnnModel::new(Path::new(KOKORO_MODEL), Path::new(KOKORO_VOICES), espeak)
        .unwrap_or_else(|e| panic!("load kokoro: {e}"));
    let sid = *model
        .voices()
        .and_then(|v| v.get(kokoro_voice))
        .unwrap_or_else(|| panic!("kokoro voice `{kokoro_voice}` not found"));
    for (idx, text) in texts.iter().take(SENTENCES_PER_SPEAKER).enumerate() {
        let (samples, sr) = model.synthesize(text, Some(sid), None).unwrap();
        let phonemes = model.phonemize(text).unwrap().chars().filter(|c| c.is_alphabetic()).count();
        let file = format!("{name}__0__{idx}.wav");
        write_wav(&out_dir.join(&file), &samples, sr);
        writeln!(manifest, "{name}\t0\t{espeak}\t{idx}\t{file}\t{sr}\t{phonemes}\t{}", text.replace('\t', " ")).unwrap();
    }
    eprintln!("rendered {name}");
}

fn render_other(
    name: &str,
    engine: &OtherEngine,
    espeak: &str,
    sentences: &HashMap<String, Vec<String>>,
    out_dir: &Path,
    manifest: &mut std::fs::File,
) {
    let texts = sentences_for(sentences, espeak)
        .unwrap_or_else(|| panic!("no sentences for `{espeak}` ({name})"));
    let cpu = &piper_rs::Backend::Cpu;
    let mut synthesize: Box<dyn FnMut(&str) -> (Vec<f32>, u32)> = match engine {
        OtherEngine::Mms { model, tokens } => {
            let mut m = MmsModel::new(model, tokens, espeak, cpu).unwrap();
            Box::new(move |t| m.synthesize(t, None, None).unwrap())
        }
        OtherEngine::Coqui { model, config } => {
            let mut m = CoquiVitsModel::new(model, config, espeak, cpu).unwrap();
            Box::new(move |t| m.synthesize(t, None, None).unwrap())
        }
        OtherEngine::GlowTts { model, vocoder, lexicon } => {
            let mut m = GlowTtsHifiganModel::new(model, vocoder, lexicon, cpu).unwrap();
            Box::new(move |t| m.synthesize(t, None, None).unwrap())
        }
        OtherEngine::Mimic3 { config } => {
            let mnn = PathBuf::from(config.to_string_lossy().replace(".onnx.json", ".mnn"));
            let mut m = PiperModel::from_mimic3(&mnn, config, cpu).unwrap();
            Box::new(move |t| m.synthesize(t, None).unwrap())
        }
        OtherEngine::KokoroJa { voice, dict } => {
            let mut m = KokoroMnnModel::new(Path::new(KOKORO_MODEL), Path::new(KOKORO_VOICES), "ja").unwrap();
            m.load_japanese_dict(&dict.to_string_lossy()).unwrap();
            let sid = *m.voices().and_then(|v| v.get(voice)).expect("kokoro voice");
            Box::new(move |t| m.synthesize(t, Some(sid), None).unwrap())
        }
    };
    for (idx, text) in texts.iter().take(SENTENCES_PER_SPEAKER).enumerate() {
        let (samples, sr) = synthesize(text);
        let letters = text.chars().filter(|c| c.is_alphabetic()).count();
        let file = format!("{name}__0__{idx}.wav");
        write_wav(&out_dir.join(&file), &samples, sr);
        writeln!(manifest, "{name}\t0\t{espeak}\t{idx}\t{file}\t{sr}\t{letters}\t{}", text.replace('\t', " ")).unwrap();
    }
    eprintln!("rendered {name}");
}

fn load_sentences(path: &str) -> HashMap<String, Vec<String>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|line| {
            let mut cols = line.splitn(3, '\t');
            let run = cols.next()?;
            let _app_lang = cols.next()?;
            let text = cols.next()?;
            Some((run.to_string(), text.to_string()))
        })
        .fold(HashMap::new(), |mut acc, (run, text)| {
            acc.entry(run).or_insert_with(Vec::new).push(text);
            acc
        })
}

// Short sentences keep each sample near the 5 s a listener needs.
fn sentences_for(sentences: &HashMap<String, Vec<String>>, espeak: &str) -> Option<Vec<String>> {
    let run = match espeak {
        "zh" => "cmn",
        other => other,
    };
    let base = run.split('-').next().unwrap_or(run);
    let texts = sentences.get(run).or_else(|| sentences.get(base))?;
    let mut by_length: Vec<String> = texts.clone();
    by_length.sort_by_key(|t| t.chars().count());
    let median = by_length.len() / 4;
    Some(by_length.into_iter().skip(median).collect())
}

fn pick_speakers(map: &HashMap<String, i64>) -> Vec<(String, Option<i64>)> {
    if map.is_empty() {
        return vec![("0".to_string(), None)];
    }
    let mut speakers: Vec<(&String, &i64)> = map.iter().collect();
    speakers.sort_by_key(|(_, id)| **id);
    let step = (speakers.len() / MAX_SPEAKERS_PER_MODEL).max(1);
    speakers
        .into_iter()
        .step_by(step)
        .take(MAX_SPEAKERS_PER_MODEL)
        .map(|(name, id)| (name.replace(['/', ' '], "_"), Some(*id)))
        .collect()
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) {
    let data: Vec<u8> = samples
        .iter()
        .flat_map(|&s| ((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes())
        .collect();
    let mut w = std::fs::File::create(path).unwrap();
    w.write_all(b"RIFF").unwrap();
    w.write_all(&(36 + data.len() as u32).to_le_bytes()).unwrap();
    w.write_all(b"WAVEfmt ").unwrap();
    w.write_all(&16u32.to_le_bytes()).unwrap();
    w.write_all(&1u16.to_le_bytes()).unwrap();
    w.write_all(&1u16.to_le_bytes()).unwrap();
    w.write_all(&sample_rate.to_le_bytes()).unwrap();
    w.write_all(&(sample_rate * 2).to_le_bytes()).unwrap();
    w.write_all(&2u16.to_le_bytes()).unwrap();
    w.write_all(&16u16.to_le_bytes()).unwrap();
    w.write_all(b"data").unwrap();
    w.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
    w.write_all(&data).unwrap();
}
