#!/usr/bin/env bash
# Runs locally: pull small text logs back from the box (no audio, checkpoints or tensorboard dirs).
# Usage: fetch_logs.sh <ssh_host> <ssh_port> <dest_dir>
set -euo pipefail
HOST=$1 PORT=$2 DEST=$3
mkdir -p "$DEST"
ssh -i ~/.ssh/personalkey -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p "$PORT" "root@$HOST" \
  'cd /root/calib && tar czf - logs dataset/summary.json dataset/config.json \
     $(find runs -maxdepth 2 -type f \( -name "*.csv" -o -name "*.txt" -o -name "*.json" -o -name "train.log" \)) 2>/dev/null' \
  | tar xzf - -C "$DEST"
du -sh "$DEST"
