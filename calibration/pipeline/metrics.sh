#!/usr/bin/env bash
# metrics.sh RUN_JOB_DIR: perf rows and loss rows from the newest metrics.csv
f=$(ls -t $1/logs/*/metrics.csv | head -1)
python3 - "$f" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    if not r.get("perf/steps_per_s"):
        continue
    losses = " ".join(f"{k[6:]}={float(v):.3f}" for k, v in r.items() if k.startswith("train_") and k != "train_reversal" and v)
    print(r["step"], "sps", r["perf/steps_per_s"][:5], "samp/s", r["perf/samples_per_s"][:5], "wait", r["perf/data_wait_share"][:5], "peakGB", r["perf/gpu_peak_gb"][:4], losses)
PY
