#!/bin/bash
# Final stage, as ONE job: the GPU part (failure analysis + student videos) and
# the CPU part (environment capture, aggregation, figures, tables, report.pdf)
# run inside the same reservation, so no separate GPU is ever held for CPU work.
set -uo pipefail
. /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci645-hw2/scripts/env.sh
cd "$PROJ"
echo "[finalize] host=$(hostname -s) $(date -Is)"

echo "===== failure analysis + student rollout videos (GPU) ====="
"$PY" scripts/stage2b_failure.py --condition "${COND:-act_clean_n100_s0}" --n_eval 100
rc=$?; [ $rc -ne 0 ] && echo "[finalize] failure analysis rc=$rc (continuing)"

echo "===== environment capture ====="
CUDA_VISIBLE_DEVICES= "$PY" scripts/capture_env.py | tail -3

echo "===== aggregate + figures + tables + report.pdf (CPU) ====="
CUDA_VISIBLE_DEVICES= "$PY" scripts/stage3_report.py --teacher_dir "$PROJ/runs/teacher"
rc=$?
echo "[finalize] stage3 rc=$rc"
exit $rc
