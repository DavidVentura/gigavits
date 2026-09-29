#!/usr/bin/env bash
# Smoke: init -> warmup (300 steps) -> render step-0 check WAVs.
set -uo pipefail
cd /workspace/gigapiper/train
# the container sees 255 CPUs but has a 30-core quota; torch would start 255 threads
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
T=/workspace/gigapiper/calibration/pipeline/tm.sh
C=/workspace/gigapiper/calibration/pipeline/smoke_train.json
R=/workspace/runs/smoke
PY=.venv/bin/python
v() { find /workspace/bucket/tts -name "$1.onnx" | head -1; }
: $T smoke-train init $PY -m gigatrain init --config $C --run-dir $R --base-checkpoint /workspace/ckpt/lessac.ckpt \
  --voice en-gb=$(v en_GB-alan-medium) --voice es=$(v es_ES-sharvard-medium) --voice ru=$(v ru_RU-dmitri-medium) \
  --voice pl=$(v pl_PL-gosia-medium) --voice it=$(v es_ES-sharvard-medium) || exit 1
$T smoke-train render-init $PY -m gigatrain render --config $C --run-dir $R --weights $R/init.pt --out $R/render-init || exit 1
$T smoke-train warmup-300 $PY -m gigatrain warmup --config $C --run-dir $R || exit 1
$T smoke-train render-warmup $PY -m gigatrain render --config $C --run-dir $R --weights $R/warmup.pt --out $R/render-warmup || exit 1
echo SMOKE_A_DONE
