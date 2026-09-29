#!/usr/bin/env bash
# Smoke: resume 2000 -> 2300, export ONNX, render at the last checkpoint, score step-0 and step-2k renders.
set -uo pipefail
cd /workspace/gigapiper/train
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/workspace/gigapiper/calibration/pipeline
T=$P/tm.sh
C=$P/smoke_train.json
R=/workspace/runs/smoke
PY=.venv/bin/python
cp $R/backend/checkpoints/last.ckpt $R/step2000.ckpt
$T smoke-train resume-2000-2300 $PY -m gigatrain train --config $C --run-dir $R --resume --max-steps 2300 > $R/resume.log 2>&1 || { echo RESUME_FAILED; exit 1; }
$T smoke-train export-onnx $PY -m gigatrain export --checkpoint $R/backend/checkpoints/last.ckpt --out $R/export > $R/export.log 2>&1 || { echo EXPORT_FAILED; tail -20 $R/export.log; }
$T smoke-train render-2300 $PY -m gigatrain render --config $C --run-dir $R --weights $R/backend/checkpoints/last.ckpt --out $R/render-2300 || exit 1
S=/workspace/teacher-smoke/shards/shard-00000
cd /workspace/gigapiper/eval
for d in render-init render-warmup render-2300; do
  ../train/.venv/bin/python $P/eval_bridge.py $R/$d $S
  $T smoke-eval score-student-$d $P/eval_env.sh venv/bin/python score.py --config $P/eval_box.toml --manifest $R/$d/student.tsv --reference teacher=$R/$d/teacher.tsv --out $R/$d/score-student > $R/$d/score.log 2>&1 || { echo SCORE_FAILED $d; tail -20 $R/$d/score.log; }
done
$T smoke-eval score-teacher $P/eval_env.sh venv/bin/python score.py --config $P/eval_box.toml --manifest $R/render-2300/teacher.tsv --reference teacher=$R/render-2300/teacher.tsv --out $R/render-2300/score-teacher > $R/render-2300/score-teacher.log 2>&1 || { echo SCORE_FAILED teacher; tail -20 $R/render-2300/score-teacher.log; }
echo SMOKE_C_DONE
