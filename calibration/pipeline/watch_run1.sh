#!/usr/bin/env bash
# Runs beside run 1 training: weights-only copies every 10k steps, evaluation at 25k and 50k,
# pruning of kept checkpoints (last two plus the evaluation ones). Outputs to $R/pull for the laptop.
set -uo pipefail
P=/workspace/gigapiper/calibration/pipeline
T=$P/tm.sh
R=/workspace/runs/run1
C=$R/run1.json
K=$R/backend/checkpoints
PULL=$R/pull
mkdir -p $PULL
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
done_file=$R/watch.done
touch $done_file

evaluate() {
  local step=$1 ckpt=$2
  local E=$R/eval-$step
  mkdir -p $E
  cd /workspace/gigapiper/train
  $T run1-eval export-$step .venv/bin/python -m gigatrain export --checkpoint $ckpt --out $E/onnx \
    --language en-gb --language es --language pl --language ru --language it > $E/export.log 2>&1
  $T run1-eval render-$step env PYTHONPATH=. .venv/bin/python $P/render_eval.py --config $C --run-dir $R --weights $ckpt \
    --held-out $R/held_out.jsonl --out $E/render --device cpu > $E/render.log 2>&1 || return
  cd /workspace/gigapiper/eval
  PYTHONPATH=. venv/bin/python $P/filter_per.py $E/render/student.tsv > $E/filter_per.log 2>&1
  $T run1-eval score-$step $P/eval_env.sh venv/bin/python score.py --config $P/eval_run1_cpu.toml --manifest $E/render/student.tsv \
    --reference teacher=$E/render/teacher.tsv --out $E/score > $E/score.log 2>&1
  if [ ! -f $R/eval-teacher/score/summary.csv ]; then
    $T run1-eval score-teacher $P/eval_env.sh venv/bin/python score.py --config $P/eval_cpu.toml --manifest $E/render/teacher.tsv \
      --reference teacher=$E/render/teacher.tsv --out $R/eval-teacher/score > $E/score-teacher.log 2>&1
    mkdir -p $PULL/eval-teacher && cp $R/eval-teacher/score/*.csv $PULL/eval-teacher/
  fi
  # small listening set: every render as Opus
  mkdir -p $PULL/eval-$step/opus
  for f in $E/render/*/*.wav; do
    l=$(basename $(dirname $f))
    ffmpeg -loglevel error -y -i $f -c:a libopus -b:a 24k $PULL/eval-$step/opus/$l-$(basename ${f%.wav}).opus
  done
  cp $E/score/*.csv $E/render/student.tsv $PULL/eval-$step/ 2>/dev/null
  cp $E/export.log $PULL/eval-$step/
}

while true; do
  for ckpt in $(ls $K/epoch=*.ckpt 2>/dev/null | sort -t= -k2 -n); do
    epoch=$(basename $ckpt .ckpt | cut -d= -f2)
    step=$(( (epoch + 1) * 1000 ))
    grep -qx "$step" $done_file && continue
    sleep 30  # let the writer finish
    if (( step % 10000 == 0 )); then
      $T run1-pull weights-$step /workspace/gigapiper/train/.venv/bin/python -c "
import torch, sys
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in ck['state_dict'].items() if k.startswith('model.')}
torch.save({'state_dict': state, 'hyper_parameters': ck['hyper_parameters'], 'global_step': ck.get('global_step')}, sys.argv[2])
" $ckpt $PULL/weights-$step.pt
    fi
    if (( step == 25000 || step == 50000 || step == 100000 )); then
      evaluate $step $ckpt
      ln -f $ckpt $PULL/full-$step.ckpt
    fi
    echo $step >> $done_file
  done
  # keep the newest two periodic checkpoints and the evaluation ones
  ls $K/epoch=*.ckpt 2>/dev/null | sort -t= -k2 -n | head -n -2 | grep -v -E 'epoch=(24|49|99)\.ckpt' | xargs -r rm -f
  sleep 60
done
