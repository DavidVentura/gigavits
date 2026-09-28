"""Speaker tiers, recording-condition tags and sampling weights from the owner's verdicts."""
import csv
from pathlib import Path

# Character voices and voices the owner heard as broken: never teachers.
EXCLUDED = {
    "de_DE-glados-medium",
    "sv_SE-lisa-medium",
    # Trained on the old 130-symbol map, which drops every Vietnamese tone digit except ɜ:
    # their tones are guesses, which a listener who does not speak Vietnamese cannot hear.
    "vi_VN-vivos-x_low",
    "vi_VN-25hours_single-low",
}

# Relative sampling weights; the lineup gets its own weight when it is chosen.
WEIGHTS = {"good": 1.0, "natural_degraded": 0.5, "clean_robot": 0.3}


def tier(human: str, audio: str) -> str:
    return {
        ("pass", "pass"): "good",
        ("pass", "fail"): "natural_degraded",
        ("fail", "pass"): "clean_robot",
        ("fail", "fail"): "excluded",
    }[(human, audio)]


def condition(sample_rate: int, bandwidth_khz: float, audio: str) -> str:
    if audio == "fail":
        return "degraded"
    if sample_rate < 22050 or bandwidth_khz < 8.5:
        return "narrow_band"
    return "clean"


def main() -> None:
    here = Path(__file__).parent
    with (here / "metrics.csv").open() as f:
        metrics = {(r["voice"], r["speaker"]): r for r in csv.DictReader(f)}
    # Candidate corpora (voices/candidates/<lang>/) carry no signal metrics; their manifests give the
    # espeak voice and sample rate, which is all the tiering needs from them.
    manifest_rows = {}
    for manifest in [here / "samples" / "manifest.tsv", *sorted((here / "candidates").glob("*/manifest.tsv"))]:
        with manifest.open() as f:
            for r in csv.DictReader(f, delimiter="\t"):
                manifest_rows[(r["voice"], r["speaker"])] = r
    with (here / "verdicts.csv").open() as f:
        verdicts = list(csv.DictReader(f))
    # Re-listen corrections live in their own file so a fresh export from the page cannot undo them.
    with (here / "verdict_overrides.csv").open() as f:
        overrides = {(o["voice"], o["speaker"]): o for o in csv.DictReader(f)}
    for v in verdicts:
        o = overrides.get((v["voice"], v["speaker"]))
        if o is None:
            continue
        v["human"] = o["human"] or v["human"]
        v["audio"] = o["audio"] or v["audio"]

    rows = []
    for v in verdicts:
        key = (v["voice"], v["speaker"])
        m = metrics.get(key)
        source = manifest_rows[key]
        # Kokoro voices are optional synthetic teachers: only the ones that pass both questions are used.
        rejected_kokoro = v["voice"].startswith("kokoro-") and (v["human"], v["audio"]) != ("pass", "pass")
        t = "excluded" if v["voice"] in EXCLUDED or rejected_kokoro else tier(v["human"], v["audio"])
        rows.append({
            "lang": v["lang"],
            "espeak": source["espeak"],
            "voice": v["voice"],
            "speaker": v["speaker"],
            "human": v["human"],
            "audio": v["audio"],
            "tier": t,
            "condition": condition(
                int(source["sample_rate"]), float(m["bandwidth_khz"]) if m else 11.0, v["audio"]
            ),
            "weight": WEIGHTS.get(t, 0.0),
            "articulation_rate": m["articulation_rate"] if m else "",
        })
    with (here / "tiers.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["lang"], r["tier"], r["voice"], r["speaker"])))
    print(f"{len(rows)} speakers -> tiers.csv")


if __name__ == "__main__":
    main()
