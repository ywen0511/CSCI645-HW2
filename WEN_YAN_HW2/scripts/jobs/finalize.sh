#!/bin/bash
# GPU job (after students.sh and teacher.sh): frozen fresh suite B, failure
# analysis + verified videos, diagnostics, environment capture, checkpoint
# bundles. Independent steps run concurrently on separate GPUs.
set -uo pipefail
J="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$J/../env.sh"; cd "$PROJ"
L="$HW2_RUNS/logs/finalize"; mkdir -p "$L"
echo "[finalize] $(hostname -s) $(date -Is)"
"$PY" scripts/capture_env.py > "$L/capture_env.log" 2>&1 || echo "capture_env FAILED"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/stage2_eval.py --suite B > "$L/suiteB.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/stage2b_failure.py > "$L/failure.log" 2>&1 & p2=$!
CUDA_VISIBLE_DEVICES=2 "$PY" scripts/diag_student.py > "$L/diag_student.log" 2>&1 & p3=$!
CUDA_VISIBLE_DEVICES=3 "$PY" scripts/diag_replay.py > "$L/diag_replay.log" 2>&1 & p4=$!
rc=0
for p in $p1 $p2 $p3 $p4; do wait $p || rc=1; done
for f in suiteB failure diag_student diag_replay; do echo "== $f"; grep -v "^\s*$" "$L/$f.log" | grep -v "Device or resource busy\|^Traceback\|File \"/" | tail -n 25; done
"$PY" scripts/collect_run_artifacts.py --bundle > "$L/collect.log" 2>&1 || rc=1
tail -n 6 "$L/collect.log"
echo "FINALIZE_JOB_DONE rc=$rc"
exit $rc
