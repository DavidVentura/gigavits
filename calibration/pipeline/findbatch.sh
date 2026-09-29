#!/usr/bin/env bash
cd /workspace/gigapiper/train
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
T=/workspace/gigapiper/calibration/pipeline/tm.sh
$T smoke-train find-batch-nobench .venv/bin/python -m gigatrain find-batch --config /workspace/gigapiper/calibration/pipeline/smoke_train.json --run-dir /workspace/runs/smoke --sizes 16,32,48 --steps 40 2>&1 | grep -E "^batch|Error|error"
$T smoke-train find-batch-bench .venv/bin/python -m gigatrain find-batch --config /workspace/gigapiper/calibration/pipeline/ab_bench.json --run-dir /workspace/runs/smoke --sizes 32 --steps 40 2>&1 | grep -E "^batch|Error|error"
echo FB_DONE
