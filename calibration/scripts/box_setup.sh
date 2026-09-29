#!/usr/bin/env bash
# Runs on the rented box. Installs piper1-gpl training, measures PyPI/HF download speed,
# downloads voices + lessac checkpoint. Writes logs to /root/calib/logs.
set -euo pipefail
PIPER_COMMIT=efffbfb226bfb511ebbcf55d0cecd8b35a89743d
W=/root/calib
L=$W/logs
mkdir -p $L $W/voices
cd $W
stamp() { echo "$(date +%s) $1" >> $L/setup_timeline.txt; }
stamp setup_start

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential cmake ninja-build git sysstat curl time >/dev/null
stamp apt_done

pip_speed() {
  local spec=$1
  rm -rf /tmp/pipdl
  local t0=$(date +%s.%N)
  pip download -q --no-deps --no-cache-dir -d /tmp/pipdl "$spec"
  local t1=$(date +%s.%N)
  local bytes=$(du -sb /tmp/pipdl | cut -f1)
  python3 -c "print('pypi $spec bytes=$bytes seconds=%.2f MBps=%.2f' % ($t1-$t0, $bytes/1e6/($t1-$t0)))" | tee -a $L/download_speed.txt
}
pip_speed scipy==1.15.3
pip_speed onnxruntime==1.22.0
stamp pypi_test_done

hf_get() {
  local url=$1 out=$2
  curl -sSL --fail -o "$out" -w "hf $(basename "$out") bytes=%{size_download} seconds=%{time_total} MBps=%{speed_download}\n" "$url" \
    | awk '{ for (i=1;i<=NF;i++) if ($i ~ /^MBps=/) { split($i,a,"="); $i=sprintf("MBps=%.2f", a[2]/1e6) } print }' | tee -a $L/download_speed.txt
}
HF=https://huggingface.co
hf_get "$HF/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt" $W/lessac.ckpt
stamp hf_ckpt_done
for v in en/en_GB/alan/medium/en_GB-alan-medium en/en_GB/alba/medium/en_GB-alba-medium \
         es/es_ES/sharvard/medium/es_ES-sharvard-medium de/de_DE/thorsten/medium/de_DE-thorsten-medium; do
  hf_get "$HF/rhasspy/piper-voices/resolve/main/$v.onnx" $W/voices/$(basename $v).onnx
  hf_get "$HF/rhasspy/piper-voices/resolve/main/$v.onnx.json" $W/voices/$(basename $v).onnx.json
done
stamp hf_voices_done

git clone -q https://github.com/OHF-Voice/piper1-gpl $W/piper1-gpl
cd $W/piper1-gpl
git checkout -q $PIPER_COMMIT
t0=$(date +%s)
pip install -q -e '.[train]' scikit-build 2>&1 | tail -5
echo "pip install -e .[train] seconds=$(( $(date +%s) - t0 ))" | tee -a $L/download_speed.txt
./build_monotonic_align.sh
python3 setup.py -q build_ext --inplace > $L/build_ext.log 2>&1
stamp piper_install_done

{
  echo "piper1-gpl commit $(git rev-parse HEAD)"
  python3 -c "import sys, torch, lightning, onnxruntime, onnx, librosa; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version()); print('lightning', lightning.__version__); print('onnxruntime', onnxruntime.__version__); print('onnx', onnx.__version__); print('librosa', librosa.__version__)"
} | tee $L/versions.txt
pip freeze > $L/pip_freeze.txt
python3 -c "from piper.train.vits import monotonic_align; from piper import espeakbridge; print('piper train imports ok')"
stamp setup_done
