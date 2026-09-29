"""Student (held-out renders) vs teacher (training clips) per voice and overall: CER, LID, UTMOS,
plus the student's PER and speaker similarity.  python compare_eval.py student_summary.csv teacher_summary.csv"""
import csv
import statistics as st
import sys


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


student = {(r["voice"], r["speaker"]): r for r in csv.DictReader(open(sys.argv[1]))}
teacher = {(r["voice"], r["speaker"]): r for r in csv.DictReader(open(sys.argv[2]))}
keys = sorted(student)
print("metric              student_mean student_median teacher_mean teacher_median (same voices)")
for k in ("cer", "lid_target", "UTMOS", "per", "speaker_similarity"):
    s = [f(student[v][k]) for v in keys if f(student[v][k]) is not None]
    t = [f(teacher[v][k]) for v in keys if v in teacher and f(teacher[v].get(k)) is not None]
    ts = f"{st.mean(t):.3f} {st.median(t):.3f}" if t else "-"
    print(f"{k:20s} {st.mean(s):.3f} {st.median(s):.3f}   {ts}   (n={len(s)})")
low = sorted(keys, key=lambda v: f(student[v]["speaker_similarity"]) or 0)
print("lowest speaker similarity:", ", ".join(f"{v[0]}.{v[1]}={f(student[v]['speaker_similarity']):.2f}" for v in low[:8]))
print("highest:", ", ".join(f"{v[0]}.{v[1]}={f(student[v]['speaker_similarity']):.2f}" for v in low[-6:]))
print("share with speaker similarity >= 0.7:", round(sum((f(student[v]["speaker_similarity"]) or 0) >= 0.7 for v in keys) / len(keys), 3))
