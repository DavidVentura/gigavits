#!/usr/bin/env bash
# Laptop: upload the repo to /workspace/gigapiper on a box. The tracked files are not enough: the app
# catalogs, the selected sentence lists and espeak-ng-data are untracked inputs the box configs read.
# The box gets the laptop's sentence lists rather than reselecting them, so its held-out sentences are
# exactly the ones the laptop's evaluation uses.
# upload.sh [env file in this directory, default box.env]
set -euo pipefail
D=$(dirname "$(readlink -f "$0")")
source "$D/${1:-box.env}"
cd "$D/../.."
E="ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR -p $SSH_PORT"
{
  git ls-files
  printf '%s\n' generate/data/source_catalog.json generate/data/mnn_catalog.json
  find generate/sentences -name '*.txt'
} | rsync -az --files-from=- -e "$E" ./ root@"$SSH_HOST":/workspace/gigapiper/
rsync -az -e "$E" coverage/work/espeak-ng-data root@"$SSH_HOST":/workspace/gigapiper/coverage/work/
