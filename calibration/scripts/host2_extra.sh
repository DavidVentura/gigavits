#!/usr/bin/env bash
# Host 2 follow-up after host2_all.sh: torch.compile and a length-capped larger batch.
S=/root/calib/scripts
W=/root/calib
while pgrep -f "^bash scripts/host2_all.sh" > /dev/null; do sleep 10; done
echo "$(date +%s) host2_extra_start" >> $W/logs/phase_times.txt
$S/run_one.sh h2_bf16_bs32_w8_compile bf16-mixed 32 8 12 --compile
python3 - <<PY
import csv, wave
from pathlib import Path
W = Path("$W")
rows = list(csv.reader(open(W / "dataset/metadata.csv", encoding="utf-8"), delimiter="|"))
def seconds(name):
    with wave.open(str(W / "dataset/wavs" / name)) as w:
        return w.getnframes() / w.getframerate()
kept = [r for r in rows if seconds(r[0]) <= 12.0]
with open(W / "dataset/metadata_max12s.csv", "w", encoding="utf-8", newline="") as f:
    csv.writer(f, delimiter="|").writerows(kept)
PY
CSV=$W/dataset/metadata_max12s.csv CACHE=$W/cache_max12s $S/run_one.sh h2_max12s_bf16_bs48_w8 bf16-mixed 48 8 10
python3 $S/analyze.py $W/runs $W/logs/results_host2.csv > $W/logs/analyze_host2.txt 2>&1
echo "$(date +%s) host2_extra_done" >> $W/logs/phase_times.txt
