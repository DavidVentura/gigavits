#!/usr/bin/env bash
# Runs after queue_b: length-capped dataset (<= 12 s) for larger batches, then the 55-front-end memory test.
S=/root/calib/scripts
W=/root/calib
stamp() { echo "$(date +%s) $1" >> $W/logs/phase_times.txt; }
while pgrep -f queue_b.sh > /dev/null; do sleep 10; done
stamp queue_c_start
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
print(len(rows), "->", len(kept))
PY
OMP_NUM_THREADS=4 $S/run_one.sh g1_bf16_bs32_w8_omp4 bf16-mixed 32 8 10
export CSV=$W/dataset/metadata_max12s.csv CACHE=$W/cache_max12s
$S/run_one.sh f0_max12s_fp32_bs32_w8 32-true 32 8 10
$S/run_one.sh f1_max12s_bf16_bs32_w8 bf16-mixed 32 8 10
$S/run_one.sh f2_max12s_bf16_bs48_w8 bf16-mixed 48 8 10
$S/run_one.sh f3_max12s_bf16_bs64_w12 bf16-mixed 64 12 10
unset CSV CACHE
stamp mem55_start
cd $W/piper1-gpl
for args in "1 32 bf16" "55 16 bf16" "55 32 bf16" "55 32 fp32"; do
  python3 $S/mem55.py $args >> $W/logs/mem55.jsonl 2>> $W/logs/mem55.err || echo "{\"failed\": \"$args\"}" >> $W/logs/mem55.jsonl
done
stamp queue_c_done
