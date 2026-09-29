#!/usr/bin/env bash
# Laptop: pull run 1's small outputs (reports, Opus, weights-only copies, full checkpoints at 25k/50k, logs).
source "$(dirname "$0")/box.env"
D=/home/david/git/gigapiper/runs/run1
E="ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR -p $SSH_PORT"
rsync -az --partial -e "$E" root@$SSH_HOST:/workspace/runs/run1/pull/ $D/
rsync -az -e "$E" --include='*.txt' --include='*.args' --include='*.json' --include='*.jsonl' --include='*.log' --include='metrics.csv' --include='hparams.yaml' --include='*/' --exclude='*.ckpt' --exclude='*.pt' --exclude='*.wav' --exclude='*' \
  --prune-empty-dirs root@$SSH_HOST:/workspace/runs/run1/ $D/box/
rsync -az -e "$E" root@$SSH_HOST:/workspace/timings.csv $D/box/timings.csv

# Laptop disk is limited: keep the two latest weights-only copies plus every 50k milestone, and the
# latest full checkpoint plus the 50k gate one. Pruned files are removed from the box's pull/ staging
# folder too, or the next rsync would fetch them again; the box's own training checkpoints are untouched.
prune=$(python3 - "$D" <<'PY'
import re, sys
from pathlib import Path
d = Path(sys.argv[1])
def steps(pattern):
    return sorted(int(m.group(1)) for p in d.iterdir() if (m := re.fullmatch(pattern, p.name)))
weights = steps(r"weights-(\d+)\.pt")
fulls = steps(r"full-(\d+)\.ckpt")
keep_w = set(weights[-2:]) | {s for s in weights if s % 50000 == 0}
keep_f = set(fulls[-1:]) | {50000}
drop = [f"weights-{s}.pt" for s in weights if s not in keep_w] + [f"full-{s}.ckpt" for s in fulls if s not in keep_f]
print(" ".join(drop))
PY
)
for f in $prune; do
  rm -f "$D/$f"
  $E root@$SSH_HOST rm -f "/workspace/runs/run1/pull/$f"
  echo "$(date -Is) pruned $f"
done
