#!/usr/bin/env bash
# Training-only host: system packages, uv, train venv (CUDA 13.0 torch), MAS kernel, eval venv (+ CUDA 12 cuBLAS/cuDNN for faster-whisper).
set -uo pipefail
P=/workspace/gigapiper/calibration/pipeline
TM=$P/tm.sh
export DEBIAN_FRONTEND=noninteractive
$TM setup2 apt bash -c 'apt-get update -qq && apt-get install -y -qq build-essential git curl rsync bc libsndfile1 ffmpeg >/dev/null' || exit 1
$TM setup2 uv bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null' || exit 1
export PATH=$HOME/.local/bin:$PATH
IDX=https://download.pytorch.org/whl/cu130
$TM setup2 venv-train bash -c "cd /workspace/gigapiper/train && uv venv -q -p 3.12 .venv && grep -v '^torch==' requirements.txt > /tmp/train-req.txt && uv pip install -q -p .venv/bin/python 'torch==2.13.0' --index-url $IDX && uv pip install -q -p .venv/bin/python -r /tmp/train-req.txt" || exit 1
$TM setup2 mas-kernel bash -c 'cd /workspace/gigapiper/train/gigatrain/vits/monotonic_align && ../../../.venv/bin/python setup.py -q build_ext --inplace' || exit 1
$TM setup2 venv-eval bash -c "cd /workspace/gigapiper/eval && uv venv -q -p 3.12 venv && grep -v '^torch' requirements-box.txt > /tmp/eval-req.txt && uv pip install -q -p venv/bin/python 'torch==2.13.0' --index-url $IDX && uv pip install -q -p venv/bin/python --no-deps 'torchaudio==2.11.0' --index-url $IDX && uv pip install -q -p venv/bin/python -r /tmp/eval-req.txt && uv pip install -q -p venv/bin/python nvidia-cublas-cu12 'nvidia-cudnn-cu12>=9,<10'" || exit 1
/workspace/gigapiper/train/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
echo SETUP_TRAIN_HOST_DONE
