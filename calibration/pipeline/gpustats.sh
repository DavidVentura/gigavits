#!/usr/bin/env bash
# gpustats.sh GPU_CSV SECONDS: mean/p10/p50/p90 of utilization, power and memory over the last SECONDS rows
tail -n "$2" "$1" | python3 -c "
import sys
u=[];p=[];m=[]
for line in sys.stdin:
    f=[x.strip() for x in line.split(',')]
    try: u.append(float(f[1].rstrip(' %'))); m.append(float(f[2].split()[0])); p.append(float(f[3].split()[0]))
    except (ValueError, IndexError): pass
def st(x):
    s=sorted(x); n=len(s); return f'mean {sum(s)/n:.1f} p10 {s[int(n*.1)]:.0f} p50 {s[n//2]:.0f} p90 {s[int(n*.9)]:.0f}'
print('n', len(u)); print('util %', st(u)); print('power W', st(p)); print('mem MiB', st(m))
"
