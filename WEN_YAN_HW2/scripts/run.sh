#!/bin/bash
# Run a stage script with the shared environment:  scripts/run.sh scripts/X.py [args]
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
echo "[run.sh] host=$(hostname -s) gpus=${CUDA_VISIBLE_DEVICES:-unset} $(date -Is)"
exec "$PY" "$@"
