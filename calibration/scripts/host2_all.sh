#!/usr/bin/env bash
# Host 2 (high-clock CPU) comparison: setup, dataset, a reduced set of runs, analysis.
S=/root/calib/scripts
W=/root/calib
L=$W/logs
mkdir -p $L
stamp() { echo "$(date +%s) $1" >> $L/phase_times.txt; }
stamp host2_start
bash $S/qualify.sh > /dev/null 2>&1
bash $S/box_setup.sh > $L/setup.log 2>&1 || { stamp setup_failed; exit 1; }
stamp setup_ok
cd $W
python3 $S/gen_teacher.py --sentences $S/sentences.tsv --voices-dir $W/voices --out-dir $W/dataset --per-speaker 180 --procs 16 --threads 2 > $L/gen_16proc_2threads.json
python3 $S/gen_teacher.py --sentences $S/sentences.tsv --voices-dir $W/voices --out-dir $W/bench_1p1t --speakers alan --per-speaker 20 --procs 1 --threads 1 > $L/gen_1proc_1thread.json
stamp gen_done
$S/run_one.sh h2_bf16_bs32_w8 bf16-mixed 32 8 12
$S/run_one.sh h2_fp32_bs16_w8 32-true 16 8 12
$S/run_one.sh h2_bf16_bs16_w8 bf16-mixed 16 8 12
python3 $S/analyze.py $W/runs $L/results_host2.csv > $L/analyze_host2.txt 2>&1
stamp host2_done
