"""Runs locally on the fetched logs: writes results.csv (training runs, both hosts),
teacher_generation.csv, mem55.csv, hosts.csv and extrapolation.csv.

One "step" below is one training iteration (one generator + one discriminator update on a batch).
Lightning's global_step (and piper checkpoint names) count 2 per iteration with this code.
"""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from analyze import summarize

CALIB = Path(__file__).resolve().parent.parent
LOGS = CALIB / "logs"


@dataclass(frozen=True)
class Host:
    name: str
    log_dir: Path
    instance_id: int
    offer_id: int
    gpu: str
    cpu: str
    location: str
    dollars_per_hour: float


HOSTS = [
    Host("host1", LOGS / "host1", 53269017, 45460721, "RTX 4090 24GB", "AMD EPYC 7742 (2.25 GHz, cgroup quota 61.4 cores, 256 visible)", "Netherlands", 0.4904),
    Host("host2", LOGS / "host2", 53278813, 53069200, "RTX 4090 24GB", "AMD Ryzen 9 7950X (cgroup quota 30.7 cores)", "Norway", 0.6463),
]
STEP_TARGETS = (300_000, 500_000)


def epoch(path: Path) -> int:
    return int(path.read_text().split()[0])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    runs, gens, mems, hosts, extrap = [], [], [], [], []
    for host in HOSTS:
        for run in sorted((host.log_dir / "runs").iterdir()):
            config = run.joinpath("config.txt").read_text()
            if "exit=0" not in config:
                runs.append({"host": host.name, "run": run.name, "status": "failed: " + ("OOM" if "out of memory" in run.joinpath("train.log").read_text() else "error")})
                continue
            row = summarize(run)
            row["dataset"] = "max12s" if "max12s" in config else "full"
            row["profiled_run_not_representative"] = "profile" in run.name
            runs.append({"host": host.name, "status": "ok", **row})
        for gen in sorted((host.log_dir / "logs").glob("gen_*.json")):
            gens.append({"host": host.name, "file": gen.name, **{k: v for k, v in json.loads(gen.read_text()).items() if k != "per_speaker_audio_seconds"}})
        mem_file = host.log_dir / "logs" / "mem55.jsonl"
        if mem_file.exists():
            mems += [{"host": host.name, **json.loads(line)} for line in mem_file.read_text().splitlines() if line.strip()]
        created, running, destroyed = (epoch(LOGS / f"{host.name}_{k}_epoch.txt") for k in ("create", "running", "destroy"))
        hours = (destroyed - created) / 3600
        hosts.append({
            "host": host.name, "instance_id": host.instance_id, "offer_id": host.offer_id, "gpu": host.gpu, "cpu": host.cpu,
            "location": host.location, "dollars_per_hour": host.dollars_per_hour,
            "create_to_running_min": round((running - created) / 60, 1), "billed_hours": round(hours, 2),
            "cost_dollars": round(hours * host.dollars_per_hour, 2),
        })

    ok = [r for r in runs if r["status"] == "ok" and not r["profiled_run_not_representative"]]
    rate = {h.name: h.dollars_per_hour for h in HOSTS}
    for r in ok:
        for steps in STEP_TARGETS:
            hours = steps / r["steps_per_s"] / 3600
            extrap.append({
                "host": r["host"], "run": r["run"], "precision": r["precision"], "batch_size": r["batch_size"],
                "dataset": r["dataset"], "steps_per_s": r["steps_per_s"], "target_iterations": steps,
                "hours": round(hours, 1), "dollars": round(hours * rate[r["host"]], 2),
                "hours_2_attempts": round(2 * hours, 1), "dollars_2_attempts": round(2 * hours * rate[r["host"]], 2),
                "hours_if_target_is_lightning_global_step": round(hours / 2, 1),
                "dollars_if_target_is_lightning_global_step": round(hours / 2 * rate[r["host"]], 2),
            })

    ok_keys = list(next(r for r in runs if r["status"] == "ok").keys())
    runs = [{k: r.get(k, "") for k in ok_keys} for r in runs]
    write_csv(CALIB / "results.csv", runs)
    write_csv(CALIB / "teacher_generation.csv", gens)
    if mems:
        mem_keys = list(dict.fromkeys(k for m in mems for k in m))
        write_csv(CALIB / "mem55.csv", [{k: m.get(k, "") for k in mem_keys} for m in mems])
    write_csv(CALIB / "hosts.csv", hosts)
    write_csv(CALIB / "extrapolation.csv", extrap)
    print(json.dumps(hosts, indent=1))
    print("total cost", round(sum(h["cost_dollars"] for h in hosts), 2), file=sys.stderr)


if __name__ == "__main__":
    main()
