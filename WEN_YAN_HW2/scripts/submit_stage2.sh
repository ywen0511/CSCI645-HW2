#!/bin/bash
# Launch the condition grid as 4 single-GPU jobs on gc216 (one per GPU).
# Each job trains+evaluates its shard and exits, so no GPU is held idle.
set -uo pipefail
PROJ=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci645-hw2
N=${N_SHARDS:-4}
for S in $(seq 0 $((N-1))); do
  "$PROJ/scripts/submit.sh" "s2sh$S" 1 240 8 \
      "$PROJ/scripts/run.sh" "$PROJ/scripts/stage2_run_conditions.py" \
      --shard "$S" --n_shards "$N" \
      --steps "${STEPS:-6000}" --chunk "${CHUNK:-30}" \
      --batch_size "${BS:-128}" --lr "${LR:-1e-4}" \
      --n_eval "${NEVAL:-100}" --exec_mode "${EXEC:-ensemble}"
done
