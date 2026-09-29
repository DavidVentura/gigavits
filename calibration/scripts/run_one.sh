#!/usr/bin/env bash
# Usage: run_one.sh NAME PRECISION BATCH WORKERS MINUTES [train_run.py flags...]
# One training run with GPU (nvidia-smi 1 s), CPU/RAM (sysmon 2 s, mpstat 5 s) logging.
set -uo pipefail
NAME=$1 PRECISION=$2 BS=$3 WORKERS=$4 MINUTES=$5
shift 5
W=/root/calib
D=$W/runs/$NAME
CSV=${CSV:-$W/dataset/metadata.csv}
CACHE=${CACHE:-$W/cache}
mkdir -p $D
cd $W/piper1-gpl
echo "$NAME precision=$PRECISION bs=$BS workers=$WORKERS minutes=$MINUTES flags=$* csv=$CSV" > $D/config.txt

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -l 1 > $D/gpu.csv &
NVPID=$!
S_TIME_FORMAT=ISO mpstat 5 > $D/mpstat.txt &
MPPID=$!

python3 $W/scripts/train_run.py --steplog $D/steps.csv "$@" fit \
  --data.voice_name calib \
  --data.csv_path $CSV \
  --data.audio_dir $W/dataset/wavs \
  --data.cache_dir $CACHE \
  --data.config_path $W/dataset/config.json \
  --data.dataset_type phoneme_ids \
  --data.espeak_voice en-gb-x-rp \
  --data.validation_split 0.02 \
  --data.batch_size $BS \
  --data.num_workers $WORKERS \
  --model.sample_rate 22050 \
  --model.num_speakers 5 \
  --model.mos_metric none \
  --model.warmstart_ckpt $W/lessac.ckpt \
  --trainer.precision $PRECISION \
  --trainer.max_time 00:00:${MINUTES}:00 \
  --trainer.limit_val_batches 0 \
  --trainer.num_sanity_val_steps 0 \
  --trainer.enable_checkpointing false \
  --trainer.default_root_dir $D > $D/train.log 2>&1 &
TPID=$!
python3 $W/scripts/sysmon.py $TPID $D/sysmon.csv &
SMPID=$!
wait $TPID
echo "exit=$?" >> $D/config.txt
sleep 2
kill $NVPID $MPPID $SMPID 2>/dev/null
wait 2>/dev/null
