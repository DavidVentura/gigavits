#!/usr/bin/env bash
# Smoke: warmup 300 -> render -> train to 2000 steps with GPU/CPU logging.
set -uo pipefail
cd /workspace/gigapiper/train
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
T=/workspace/gigapiper/calibration/pipeline/tm.sh
C=/workspace/gigapiper/calibration/pipeline/smoke_train.json
R=/workspace/runs/smoke
PY=.venv/bin/python
rm -rf $R/warmup
$T smoke-train warmup-300 $PY -m gigatrain warmup --config $C --run-dir $R || exit 1
$T smoke-train render-warmup $PY -m gigatrain render --config $C --run-dir $R --weights $R/warmup.pt --out $R/render-warmup || exit 1
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1 > $R/gpu.csv &
GPU=$!
vmstat -n 5 > $R/vmstat.txt &
VM=$!
$T smoke-train train-2000 $PY -m gigatrain train --config $C --run-dir $R --init $R/warmup.pt --max-steps 2000
kill $GPU $VM
echo SMOKE_B_DONE
