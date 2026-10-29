#!/bin/bash
# GPU job: teacher evaluation + dataset audit (reuses a matching trained
# teacher) concurrently with suite A, then everything in finalize.sh.
set -uo pipefail
J="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$J/../env.sh"; cd "$PROJ"
L="$HW2_RUNS/logs/evaluate_all"; mkdir -p "$L"
echo "[evaluate_all] $(hostname -s) $(date -Is)"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/stage1r_teacher.py --num_timesteps 60000000 > "$L/teacher.log" 2>&1 & pt=$!
bash "$J/parallel.sh" 4 2 scripts/stage2_eval.py --suite A; ra=$?
wait $pt; rt=$?
grep -E "^\[teacher|^\[audit|Error" "$L/teacher.log" | tail -n 20
echo "[evaluate_all] suiteA rc=$ra teacher rc=$rt $(date -Is)"
[ $ra -eq 0 ] && [ $rt -eq 0 ] || { echo "EVALUATE_ALL FAILED"; exit 1; }
bash "$J/finalize.sh"
