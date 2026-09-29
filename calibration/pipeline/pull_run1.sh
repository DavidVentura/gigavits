#!/usr/bin/env bash
# Laptop: pull run 1's small outputs (reports, Opus, weights-only copies, full checkpoints at 25k/50k, logs).
source "$(dirname "$0")/box.env"
D=/home/david/git/gigapiper/runs/run1
E="ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR -p $SSH_PORT"
rsync -az --partial -e "$E" root@$SSH_HOST:/workspace/runs/run1/pull/ $D/
rsync -az -e "$E" --include='*.txt' --include='*.args' --include='*.json' --include='*.jsonl' --include='*.log' --include='metrics.csv' --include='hparams.yaml' --include='*/' --exclude='*.ckpt' --exclude='*.pt' --exclude='*.wav' --exclude='*' \
  --prune-empty-dirs root@$SSH_HOST:/workspace/runs/run1/ $D/box/
rsync -az -e "$E" root@$SSH_HOST:/workspace/timings.csv $D/box/timings.csv
