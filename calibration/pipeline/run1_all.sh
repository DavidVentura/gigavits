#!/usr/bin/env bash
# Training host: prep (resumable) then the main run, with the checkpoint watcher beside it.
P=/workspace/gigapiper/calibration/pipeline
WARMUP_BATCH=${WARMUP_BATCH:-16} bash $P/stage_a_prep.sh || { echo PREP_FAILED; exit 1; }
setsid nohup bash $P/watch_run1.sh > /workspace/runs/run1/watch.log 2>&1 < /dev/null &
bash $P/stage_a_train.sh
