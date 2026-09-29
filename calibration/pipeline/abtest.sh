#!/usr/bin/env bash
# A/B: warmup steps/s with and without cudnn.benchmark (variable-length full-utterance batches).
cd /workspace/gigapiper/train
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
for cfg in ab_nobench ab_bench; do
  rm -rf /workspace/runs/ab-$cfg; mkdir -p /workspace/runs/ab-$cfg; cp /workspace/runs/smoke/ids.json /workspace/runs/smoke/init.pt /workspace/runs/ab-$cfg/
  t0=$(date +%s)
  .venv/bin/python -m gigatrain warmup --config /workspace/gigapiper/calibration/pipeline/$cfg.json --run-dir /workspace/runs/ab-$cfg > /workspace/ab-$cfg.log 2>&1
  echo "$cfg $(( $(date +%s) - t0 )) s"
  grep -o "perf/steps_per_s[^,]*" -r /workspace/runs/ab-$cfg/warmup/logs 2>/dev/null | head -2
done
