#!/usr/bin/env bash
# Box setup: system packages, rust, uv, piper-rs + teacher-rt, venvs (generate, train with CUDA torch, eval), MAS kernel.
set -uo pipefail
P=/workspace/gigapiper/calibration/pipeline
TM=$P/tm.sh
cd /workspace
export DEBIAN_FRONTEND=noninteractive
$TM setup apt bash -c 'apt-get update -qq && apt-get install -y -qq build-essential cmake ninja-build clang libclang-dev pkg-config git curl rsync sysstat bc autoconf automake libtool libsndfile1 ffmpeg opus-tools >/dev/null' || exit 1
$TM setup rustup bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal >/dev/null' || exit 1
$TM setup uv bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null' || exit 1
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
$TM setup clone-piper-rs bash -c 'git clone -q --recursive https://github.com/DavidVentura/piper-rs /workspace/piper-rs && cd /workspace/piper-rs && git checkout -q 764f745 && git submodule update -q --init --recursive' || exit 1
# teacher-rt builds in the background while the venvs install
( $TM setup build-teacher-rt bash -c 'cd /workspace/gigapiper/generate/rt && cargo build --release -q --target-dir /workspace/piper-rs/target' > /workspace/build-teacher-rt.log 2>&1 ) &
BUILD=$!
$TM setup venv-generate bash -c 'cd /workspace/gigapiper/generate && uv venv -q -p 3.12 .venv && uv pip install -q -p .venv/bin/python -r requirements.txt' || exit 1
$TM setup venv-train bash -c 'cd /workspace/gigapiper/train && uv venv -q -p 3.12 .venv && grep -v "^torch==" requirements.txt > /tmp/train-req.txt && uv pip install -q -p .venv/bin/python "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu128 && uv pip install -q -p .venv/bin/python -r /tmp/train-req.txt' || exit 1
$TM setup mas-kernel bash -c 'cd /workspace/gigapiper/train/gigatrain/vits/monotonic_align && ../../../.venv/bin/python setup.py -q build_ext --inplace' || exit 1
$TM setup venv-eval bash -c 'cd /workspace/gigapiper/eval && uv venv -q -p 3.12 venv && grep -v "^torch" requirements-box.txt > /tmp/eval-req.txt && uv pip install -q -p venv/bin/python "torch==2.13.0" "torchaudio" --index-url https://download.pytorch.org/whl/cu128 && uv pip install -q -p venv/bin/python -r /tmp/eval-req.txt' || exit 1
wait $BUILD
echo "build rc=$?"
ls -la /workspace/piper-rs/target/release/teacher-rt
echo SETUP_DONE
