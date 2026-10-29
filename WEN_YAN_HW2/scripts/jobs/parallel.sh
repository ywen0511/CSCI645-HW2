#!/bin/bash
# Run one stage script as N_PROC shards spread over the visible GPUs and wait
# for every shard (explicit PIDs). Exit code is non-zero if any shard failed.
#   parallel.sh <n_gpus> <procs_per_gpu> <script.py> [args...]
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
NG=$1; PPG=$2; SCRIPT=$3; shift 3
N=$((NG*PPG)); LOGD="$HW2_RUNS/logs/$(basename "$SCRIPT" .py)"; mkdir -p "$LOGD"
pids=()
for ((s=0; s<N; s++)); do
  g=$((s % NG))
  CUDA_VISIBLE_DEVICES=$g "$PY" "$PROJ/$SCRIPT" --shard $s --n_shards $N "$@" \
      > "$LOGD/shard$s.log" 2>&1 &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "[parallel] shard $i FAILED (see $LOGD/shard$i.log)"; rc=1; fi
done
grep -h "^\[RESULT\|^\[trained\|^\[skip" "$LOGD"/shard*.log | sort | tail -n 60
exit $rc
