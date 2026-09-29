#!/usr/bin/env bash
S=/root/calib/scripts
echo "$(date +%s) queue_a_start" >> /root/calib/logs/phase_times.txt
$S/run_one.sh a1_bf16_bs32_w8 bf16-mixed 32 8 12
$S/run_one.sh a2_bf16_bs64_w8 bf16-mixed 64 8 12
echo "$(date +%s) queue_a_done" >> /root/calib/logs/phase_times.txt
