#!/usr/bin/env bash
# Quick host qualification: GPU, CPU, RAM, disk, cgroup CPU quota, 30 s matmul benchmark.
set -uo pipefail
L=/root/calib/logs
mkdir -p $L
{
  echo "== nvidia-smi"; nvidia-smi
  echo "== nproc"; nproc
  echo "== cgroup cpu quota"; cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null
  echo "== lscpu"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|MHz'
  echo "== free -g"; free -g
  echo "== cgroup memory limit"; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null
  echo "== df -h"; df -h / /root 2>/dev/null
} 2>&1 | tee $L/host_info.txt
python3 /root/calib/scripts/bench_matmul.py 2>&1 | tee $L/bench_matmul.txt
