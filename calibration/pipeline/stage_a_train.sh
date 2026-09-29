#!/usr/bin/env bash
# Run 1 stage A main training (after stage_a_prep.sh) with per-second GPU logging. On a crash it
# resumes from the last epoch checkpoint; after an out-of-memory crash it continues with 8 items fewer per batch.
set -uo pipefail
P=/workspace/gigapiper/calibration/pipeline
T=$P/tm.sh
R=/workspace/runs/run1
C=$R/run1.json
cd /workspace/gigapiper/train
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1 > $R/gpu.csv 2>&1 &
echo STAGE_A_TRAIN_START
batch=${TRAIN_BATCH:-24}
start="--init $R/warmup.pt"
for attempt in 1 2 3 4 5; do
  $T stage-a-train train-attempt-$attempt .venv/bin/python -m gigatrain train --config $C --run-dir $R $start --batch-size $batch > $R/train-$attempt.log 2>&1
  rc=$?
  echo "attempt $attempt batch $batch rc=$rc"
  [ $rc -eq 0 ] && break
  grep -q -i "out of memory\|OutOfMemoryError" $R/train-$attempt.log && [ $batch -gt 16 ] && batch=$(( batch - 8 ))
  # a crash before the first epoch checkpoint starts over from the warm-up weights
  [ -f $R/backend/checkpoints/last.ckpt ] && start="--resume"
done
