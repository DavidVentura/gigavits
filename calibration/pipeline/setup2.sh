#!/usr/bin/env bash
# Second half of setup: train venv (CUDA 13.0 torch; 2.13.0 has no cu128 build), MAS kernel, eval venv.
set -uo pipefail
P=/workspace/gigapiper/calibration/pipeline
TM=$P/tm.sh
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
IDX=https://download.pytorch.org/whl/cu130
$TM setup venv-train bash -c "cd /workspace/gigapiper/train && rm -rf .venv && uv venv -q -p 3.12 .venv && grep -v '^torch==' requirements.txt > /tmp/train-req.txt && uv pip install -q -p .venv/bin/python 'torch==2.13.0' --index-url $IDX && uv pip install -q -p .venv/bin/python -r /tmp/train-req.txt" || exit 1
$TM setup mas-kernel bash -c 'cd /workspace/gigapiper/train/gigatrain/vits/monotonic_align && ../../../.venv/bin/python setup.py -q build_ext --inplace' || exit 1
$TM setup venv-eval bash -c "cd /workspace/gigapiper/eval && rm -rf venv && uv venv -q -p 3.12 venv && grep -v '^torch' requirements-box.txt > /tmp/eval-req.txt && uv pip install -q -p venv/bin/python 'torch==2.13.0' --index-url $IDX && uv pip install -q -p venv/bin/python --no-deps 'torchaudio==2.11.0' --index-url $IDX && uv pip install -q -p venv/bin/python -r /tmp/eval-req.txt" || exit 1
/workspace/gigapiper/train/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
/workspace/gigapiper/eval/venv/bin/python -c "import torch, torchaudio, speechbrain; print(torch.__version__, torchaudio.__version__, torch.cuda.is_available())"
echo SETUP2_DONE
