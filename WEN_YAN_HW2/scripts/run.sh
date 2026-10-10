#!/bin/bash
# Source the shared env, then run python with the given arguments.
set -euo pipefail
. /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci645-hw2/scripts/env.sh
echo "[run.sh] host=$(hostname -s) gpus=${CUDA_VISIBLE_DEVICES:-unset} $(date -Is)"
exec "$PY" "$@"
