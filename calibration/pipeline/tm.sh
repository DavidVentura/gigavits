#!/usr/bin/env bash
# tm.sh SECTION NAME cmd...: runs cmd, appends "section,name,seconds,exit" to /workspace/timings.csv.
section=$1; name=$2; shift 2
t0=$(date +%s.%N)
"$@"
rc=$?
t1=$(date +%s.%N)
printf '%s,%s,%.1f,%d\n' "$section" "$name" "$(echo "$t1 - $t0" | bc)" "$rc" >> /workspace/timings.csv
exit $rc
