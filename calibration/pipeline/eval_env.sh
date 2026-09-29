#!/usr/bin/env bash
# eval_env.sh cmd...: faster-whisper's CTranslate2 needs CUDA 12 cuBLAS/cuDNN next to torch's CUDA 13.
N=/workspace/gigapiper/eval/venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$N/cublas/lib:$N/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=8
exec "$@"
