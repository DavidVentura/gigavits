#!/usr/bin/env bash
# Run 1 stage A preparation: prepare -> init -> render step 0 -> warmup 3000 -> render -> held-out ids.
set -uo pipefail
P=/workspace/gigapiper/calibration/pipeline
T=$P/tm.sh
R=/workspace/runs/run1
C=$R/run1.json
mkdir -p $R
cd /workspace/gigapiper/train
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
[ -f $C ] || $PY $P/make_run1_config.py $C || exit 1
[ -f $R/ids.json ] || $T stage-a-train prepare $PY -m gigatrain prepare --config $C --run-dir $R > $R/prepare.txt 2>&1 || exit 1
[ -s $R/voices.args ] || python3 $P/choose_voices.py /workspace/plan-stage-a.txt /workspace/family.tsv $R/ids.json /workspace/bucket > $R/voices.args 2> $R/voices.txt || exit 1
[ -f $R/init.pt ] || $T stage-a-train init $PY -m gigatrain init --config $C --run-dir $R --base-checkpoint /workspace/ckpt/lessac.ckpt $(cat $R/voices.args) > $R/init.log 2>&1 || exit 1
[ -f $R/render-init/renders.tsv ] || $T stage-a-train render-init $PY -m gigatrain render --config $C --run-dir $R --weights $R/init.pt --out $R/render-init > $R/render-init.log 2>&1 || exit 1
$T stage-a-train warmup-3000 $PY -m gigatrain warmup --config $C --run-dir $R --batch-size ${WARMUP_BATCH:-32} > $R/warmup.log 2>&1 || exit 1
$T stage-a-train render-warmup $PY -m gigatrain render --config $C --run-dir $R --weights $R/warmup.pt --out $R/render-warmup > $R/render-warmup.log 2>&1 || exit 1
[ -s $R/held_out.jsonl ] || (cd /workspace/gigapiper/generate && $T stage-a-train held-out-ids env PYTHONPATH=. .venv/bin/python $P/held_out_ids.py $P/stage_a.toml $R/ids.json 5 $R/held_out.jsonl) || exit 1
echo STAGE_A_PREP_DONE
