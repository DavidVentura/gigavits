#!/usr/bin/env bash
# Usage: queue_b.sh [extra train_run flags applied to every run, e.g. --cudnn-benchmark]
S=/root/calib/scripts
F="$*"
stamp() { echo "$(date +%s) $1" >> /root/calib/logs/phase_times.txt; }
stamp queue_b_start
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $S/run_one.sh c1_bf16_bs48_w8 bf16-mixed 48 8 12 $F
$S/run_one.sh c2_fp32_bs32_w8 32-true 32 8 12 $F
$S/run_one.sh c3_fp16_bs32_w8 16-mixed 32 8 12 $F
$S/run_one.sh c4_bf16_bs16_w8 bf16-mixed 16 8 12 $F
$S/run_one.sh c5_fp32_bs16_w8 32-true 16 8 12 $F
$S/run_one.sh c6_bf16_bs32_w8_profile bf16-mixed 32 8 05 $F --profile-at 200
for w in 2 4 16 58; do
  $S/run_one.sh d_bf16_bs32_w$w bf16-mixed 32 $w 07 $F
done
$S/run_one.sh e_bf16_bs32_w8_compile bf16-mixed 32 8 12 $F --compile
stamp queue_b_done
