#!/usr/bin/env bash
# Stage A teacher generation (generate's box config without Kokoro).
set -uo pipefail
cd /workspace/gigapiper/generate
T=/workspace/gigapiper/calibration/pipeline/tm.sh
C=/workspace/gigapiper/calibration/pipeline/stage_a.toml
$T stage-a-gen all-generate-package-verify .venv/bin/python -m gigagen $C all > /workspace/stage_a_gen.log 2>&1
echo "rc=$?" >> /workspace/stage_a_gen.log
echo STAGE_A_GEN_DONE >> /workspace/stage_a_gen.log
