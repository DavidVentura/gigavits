"""Rebuilds calibration/pipeline_timings.csv from the laptop-side rows and both boxes' tm.sh logs;
prints per-section sums. Box rows keep their exit code in the command name when it was non-zero."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

out = Path("/home/david/git/gigapiper/calibration/pipeline_timings.csv")
laptop = Path("/home/david/git/gigapiper/calibration/pipeline/logs/laptop_timings.csv")
if not laptop.exists():
    laptop.write_text(out.read_text())
rows = [r for r in csv.reader(laptop.open()) if r and r[0] != "section"]
rows = [[s, c, sec, "laptop"] for s, c, sec in rows]
for host, path in (("box1-epyc7742", sys.argv[1]), ("box2-7950x", sys.argv[2])):
    for r in csv.reader(open(path)):
        section, command, seconds, rc = r
        name = command if rc == "0" else f"{command} (exit {rc})"
        rows.append([section, name, seconds, host])
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["section", "command", "seconds", "where"])
    w.writerows(rows)
sums = defaultdict(float)
for s, _, sec, _ in rows:
    sums[s] += float(sec)
for s, v in sums.items():
    print(f"{s:16s} {v/60:7.1f} min")
