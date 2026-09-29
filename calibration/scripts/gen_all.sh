#!/usr/bin/env bash
# Teacher generation benchmarks + dataset build. Times each phase into logs/phase_times.txt.
set -euo pipefail
W=/root/calib
S=$W/scripts
L=$W/logs
cd $W
stamp() { echo "$(date +%s) $1" >> $L/phase_times.txt; }
export OMP_NUM_THREADS=1

stamp durations_check_start
python3 $S/check_durations.py $W/voices $S/expose_durations.py $W/durations_models 2>&1 | tee $L/check_durations.txt
stamp durations_check_done

COMMON="--sentences $S/sentences.tsv --voices-dir $W/voices"
# 1 process, all threads (onnxruntime intra-op 32)
python3 $S/gen_teacher.py $COMMON --out-dir $W/bench_1p32t --speakers alan --per-speaker 40 --procs 1 --threads 32 > $L/gen_1proc_32threads.json
stamp gen_1p32t_done
# 1 process, 1 thread (per-core rate)
python3 $S/gen_teacher.py $COMMON --out-dir $W/bench_1p1t --speakers alan --per-speaker 20 --procs 1 --threads 1 > $L/gen_1proc_1thread.json
stamp gen_1p1t_done
# N = cores/2 processes x 2 threads: full dataset
python3 $S/gen_teacher.py $COMMON --out-dir $W/dataset --per-speaker 180 --procs 30 --threads 2 > $L/gen_30proc_2threads.json
stamp gen_dataset_done
# 60 processes x 1 thread, same work, for comparison (discarded output)
python3 $S/gen_teacher.py $COMMON --out-dir $W/bench_60p1t --per-speaker 180 --procs 60 --threads 1 > $L/gen_60proc_1thread.json
stamp gen_60p1t_done
rm -rf $W/bench_60p1t
cp $W/dataset/metadata.csv $L/dataset_metadata_head.csv.tmp && head -20 $L/dataset_metadata_head.csv.tmp > $L/dataset_metadata_head.csv && rm $L/dataset_metadata_head.csv.tmp
du -sh $W/dataset/wavs >> $L/phase_times.txt
