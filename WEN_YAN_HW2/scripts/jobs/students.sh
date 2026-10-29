#!/bin/bash
# GPU job: train all 21 students, then evaluate suite A (both modes).
set -uo pipefail
J="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$J/../env.sh"; cd "$PROJ"
echo "[students] $(hostname -s) $(date -Is) gpus=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name --format=csv,noheader
t0=$(date +%s)
bash "$J/parallel.sh" 4 2 scripts/stage2_train.py || { echo "TRAIN FAILED"; exit 1; }
echo "[students] training done in $(( $(date +%s) - t0 ))s"
t1=$(date +%s)
bash "$J/parallel.sh" 4 2 scripts/stage2_eval.py --suite A || { echo "EVAL A FAILED"; exit 1; }
echo "[students] suite A done in $(( $(date +%s) - t1 ))s"
echo STUDENTS_JOB_DONE
