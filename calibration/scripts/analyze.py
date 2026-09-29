"""Summarize training runs into a CSV (one row per run).

Steady window: from WARMUP_SECONDS after the first step end to the last step end.
Usage: analyze.py <runs_dir> <out.csv>
"""

import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

WARMUP_SECONDS = 90.0


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def gpu_rows(path: Path) -> list[tuple[float, float, float, float]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4 or "N/A" in parts:
            continue
        t = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        rows.append((t, float(parts[1]), float(parts[2]), float(parts[3])))
    return rows


def summarize(run: Path) -> dict[str, object]:
    config = dict(kv.split("=", 1) for kv in run.joinpath("config.txt").read_text().split()[1:] if "=" in kv)
    steps = read_csv(run / "steps.csv")
    t_first = float(steps[0]["t_end"])
    t0 = t_first + WARMUP_SECONDS
    window = [s for s in steps if float(s["t_end"]) >= t0]
    t_start, t_end = float(window[0]["t_end"]), float(window[-1]["t_end"])
    duration = t_end - t_start
    counted = window[1:]
    n_steps = len(counted)
    utts = sum(int(s["batch_size"]) for s in counted)
    audio = sum(float(s["audio_seconds"]) for s in counted)
    waits = [float(s["data_wait"]) for s in counted]
    compute = [float(s["t_end"]) - float(s["t_start"]) for s in counted]
    loss_g = [float(s["loss_g"]) for s in steps]

    gpu = [g for g in gpu_rows(run / "gpu.csv") if t_start <= g[0] <= t_end]
    util = [g[1] for g in gpu]
    mem_all = [g[2] for g in gpu_rows(run / "gpu.csv")]
    sysmon = [r for r in read_csv(run / "sysmon.csv") if t_start <= float(r["t"]) <= t_end]
    summary_path = run / "steps.summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    return {
        "run": run.name,
        "precision": config["precision"],
        "batch_size": int(config["bs"]),
        "num_workers": int(config["workers"]),
        "flags": config.get("flags", ""),
        "steady_seconds": round(duration, 1),
        "steady_steps": n_steps,
        "steps_per_s": round(n_steps / duration, 3),
        "utts_per_s": round(utts / duration, 1),
        "audio_sec_per_s": round(audio / duration, 1),
        "mean_step_compute_s": round(statistics.mean(compute), 4),
        "mean_data_wait_s": round(statistics.mean(waits), 4),
        "data_wait_frac": round(sum(waits) / duration, 4),
        "p90_data_wait_s": round(percentile(waits, 0.9), 4),
        "gpu_util_mean": round(statistics.mean(util), 1),
        "gpu_util_p10": percentile(util, 0.1),
        "gpu_util_p50": percentile(util, 0.5),
        "gpu_util_p90": percentile(util, 0.9),
        "gpu_power_mean_w": round(statistics.mean(g[3] for g in gpu), 1),
        "gpu_mem_used_max_mib": max(mem_all),
        "torch_max_alloc_gb": round(summary.get("max_memory_allocated_gb", float("nan")), 2),
        "torch_max_reserved_gb": round(summary.get("max_memory_reserved_gb", float("nan")), 2),
        "container_cores_used_mean": round(statistics.mean(float(r["container_cores_used"]) for r in sysmon), 2),
        "main_proc_cpu_pct_mean": round(statistics.mean(float(r["main_cpu_pct"]) for r in sysmon), 1),
        "workers_cpu_pct_mean": round(statistics.mean(float(r["children_cpu_pct"]) for r in sysmon), 1),
        "sys_mem_used_gb_max": max(float(r["mem_used_gb"]) for r in sysmon),
        "train_tree_rss_gb_max": max(float(r["tree_rss_gb"]) for r in sysmon),
        "nonfinite_loss_steps": sum(1 for x in loss_g if not math.isfinite(x)),
        "loss_g_first": loss_g[0],
        "loss_g_last_100_mean": round(statistics.mean(x for x in loss_g[-100:] if math.isfinite(x)), 3),
        "mel_last_100_mean": round(statistics.mean(float(s["train_mel"]) for s in steps[-100:]), 4),
    }


def main() -> None:
    runs_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for run in sorted(runs_dir.iterdir()):
        if not (run / "steps.csv").exists():
            continue
        try:
            rows.append(summarize(run))
        except (IndexError, ValueError, KeyError, statistics.StatisticsError) as error:
            print(f"{run.name}: cannot summarize: {error!r}", file=sys.stderr)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(
            f"{r['run']:28s} {r['precision']:10s} bs={r['batch_size']:3d} w={r['num_workers']:2d} "
            f"steps/s={r['steps_per_s']:.3f} audio_s/s={r['audio_sec_per_s']:7.1f} gpu={r['gpu_util_mean']:5.1f}% "
            f"(p10/50/90 {r['gpu_util_p10']:.0f}/{r['gpu_util_p50']:.0f}/{r['gpu_util_p90']:.0f}) "
            f"wait={r['data_wait_frac']:.3f} mem={r['torch_max_alloc_gb']}GB cores={r['container_cores_used_mean']} "
            f"main={r['main_proc_cpu_pct_mean']}% nonfinite={r['nonfinite_loss_steps']}"
        )


if __name__ == "__main__":
    main()
