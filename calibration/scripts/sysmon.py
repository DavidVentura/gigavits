"""Sample container CPU usage (cgroup), RAM, and CPU of a process tree every INTERVAL seconds.

Usage: sysmon.py <pid> <out.csv>
Columns: t, container_cores_used, mem_used_gb, main_cpu_pct, children_cpu_pct, n_children, tree_rss_gb
(cpu_pct: 100 = one core fully busy)
"""

import os
import sys
import time
from pathlib import Path

INTERVAL = 2.0
TICKS = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")


def cgroup_usage_seconds() -> float:
    v2 = Path("/sys/fs/cgroup/cpu.stat")
    if v2.exists():
        for line in v2.read_text().splitlines():
            key, value = line.split()
            if key == "usage_usec":
                return int(value) / 1e6
    return int(Path("/sys/fs/cgroup/cpuacct/cpuacct.usage").read_text()) / 1e9


def mem_used_gb() -> float:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":")
        info[key] = int(value.split()[0])
    return (info["MemTotal"] - info["MemAvailable"]) / 1e6


def proc_table() -> dict[int, tuple[int, float, float]]:
    """pid -> (ppid, cpu_seconds, rss_gb)"""
    table = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2 :].split()
        ppid = int(fields[1])
        cpu = (int(fields[11]) + int(fields[12])) / TICKS
        rss = int(fields[21]) * PAGE / 1e9
        table[int(entry)] = (ppid, cpu, rss)
    return table


def descendants(table: dict[int, tuple[int, float, float]], root: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, (ppid, _, _) in table.items():
        children.setdefault(ppid, []).append(pid)
    found, stack = set(), [root]
    while stack:
        for child in children.get(stack.pop(), []):
            found.add(child)
            stack.append(child)
    return found


def main() -> None:
    root, out = int(sys.argv[1]), Path(sys.argv[2])
    with open(out, "w", buffering=1) as f:
        f.write("t,container_cores_used,mem_used_gb,main_cpu_pct,children_cpu_pct,n_children,tree_rss_gb\n")
        prev_t, prev_cg = time.time(), cgroup_usage_seconds()
        prev_cpu: dict[int, float] = {}
        while Path(f"/proc/{root}").exists():
            time.sleep(INTERVAL)
            t, cg = time.time(), cgroup_usage_seconds()
            table = proc_table()
            if root not in table:
                break
            kids = descendants(table, root)
            dt = t - prev_t

            def delta(pid: int) -> float:
                return table[pid][1] - prev_cpu.get(pid, table[pid][1])

            main_pct = 100 * delta(root) / dt
            kids_pct = 100 * sum(delta(p) for p in kids) / dt
            rss = table[root][2] + sum(table[p][2] for p in kids)
            f.write(f"{t:.2f},{(cg - prev_cg) / dt:.2f},{mem_used_gb():.2f},{main_pct:.1f},{kids_pct:.1f},{len(kids)},{rss:.2f}\n")
            prev_t, prev_cg = t, cg
            prev_cpu = {p: table[p][1] for p in kids | {root}}


if __name__ == "__main__":
    main()
