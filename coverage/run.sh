#!/usr/bin/env bash
# Reproduces the whole pipeline. Expects work/flores200_dataset (https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz),
# work/espeak-ng-data (bucket espeak data plus the dicts it lacks from the piper-rs espeak-rs-sys 1.52 build) and work/venv with pykakasi.
set -euo pipefail
cd "$(dirname "$0")"
BUCKET=/home/david/AndroidStudioProjects/bucket/tts/1
REF=$BUCKET/en/en_US/amy/medium/en_US-amy-medium
CARGO_TARGET_DIR=/home/david/git/piper-rs/target cargo build --release -j 4
work/venv/bin/python prepare_sentences.py
/home/david/git/piper-rs/target/release/phoneme-coverage work/espeak-ng-data "$REF.onnx.json" "$REF.mnn" work/configs work/sentences.tsv work/phonemized.jsonl > work/phonemize.log 2>&1
python3 analyze.py
