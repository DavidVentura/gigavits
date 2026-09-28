use std::collections::BTreeMap;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

use piper_rs::{init_espeak, Backend, PiperModel};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct EspeakSection {
    voice: String,
}

#[derive(Deserialize, Serialize)]
struct PiperConfig {
    espeak: EspeakSection,
    #[serde(flatten)]
    rest: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize)]
struct Record<'a> {
    run: &'a str,
    idx: usize,
    phonemes: &'a str,
    tokens: &'a [String],
}

struct Args {
    espeak_data: PathBuf,
    reference_config: PathBuf,
    reference_model: PathBuf,
    work_dir: PathBuf,
    sentences_tsv: PathBuf,
    out_jsonl: PathBuf,
}

fn parse_args() -> Args {
    let usage = "usage: phoneme-coverage <espeak-data> <reference.onnx.json> <reference.mnn> <work-dir> <sentences.tsv: run\\tespeak_voice\\tsentence> <out.jsonl>";
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert!(a.len() == 6, "{usage}");
    Args {
        espeak_data: a[0].clone().into(),
        reference_config: a[1].clone().into(),
        reference_model: a[2].clone().into(),
        work_dir: a[3].clone().into(),
        sentences_tsv: a[4].clone().into(),
        out_jsonl: a[5].clone().into(),
    }
}

/// The production Piper front end reads the espeak voice from the config, so each run gets the
/// reference config with only `espeak.voice` swapped; the phoneme_id_map stays the reference one.
fn model_for_voice(args: &Args, run: &str, voice: &str) -> PiperModel {
    let reference: PiperConfig =
        serde_json::from_reader(std::fs::File::open(&args.reference_config).unwrap()).unwrap();
    let config = PiperConfig {
        espeak: EspeakSection {
            voice: voice.to_string(),
        },
        rest: reference.rest,
    };
    let config_path = args.work_dir.join(format!("{run}.onnx.json"));
    serde_json::to_writer(std::fs::File::create(&config_path).unwrap(), &config).unwrap();
    PiperModel::new(&args.reference_model, &config_path, &Backend::Cpu).unwrap()
}

fn main() {
    let args = parse_args();
    init_espeak(Path::new(&args.espeak_data)).unwrap();
    std::fs::create_dir_all(&args.work_dir).unwrap();

    let mut runs: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    let file = std::io::BufReader::new(std::fs::File::open(&args.sentences_tsv).unwrap());
    for line in file.lines() {
        let line = line.unwrap();
        let mut parts = line.splitn(3, '\t');
        let (Some(run), Some(voice), Some(sentence)) = (parts.next(), parts.next(), parts.next())
        else {
            panic!("malformed line: {line}");
        };
        runs.entry((run.to_string(), voice.to_string()))
            .or_default()
            .push(sentence.to_string());
    }

    // MNN prints device info to stdout, so records go to a file.
    let mut out = BufWriter::new(std::fs::File::create(&args.out_jsonl).unwrap());
    for ((run, voice), sentences) in &runs {
        let model = model_for_voice(&args, run, voice);
        for (idx, sentence) in sentences.iter().enumerate() {
            let debug = model.debug_frontend(sentence).unwrap();
            let record = Record {
                run,
                idx,
                phonemes: &debug.phonemes,
                tokens: &debug.tokens,
            };
            serde_json::to_writer(&mut out, &record).unwrap();
            out.write_all(b"\n").unwrap();
        }
        eprintln!("{run} ({voice}): {} sentences", sentences.len());
    }
}
