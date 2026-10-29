#!/bin/bash
# GPU job: retrain + evaluate the PPO teacher, audit datasets, record versions.
set -uo pipefail
J="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$J/../env.sh"; cd "$PROJ"
echo "[teacher] $(hostname -s) $(date -Is)"
"$PY" scripts/capture_env.py | tail -2
"$PY" scripts/stage1r_teacher.py --num_timesteps 60000000 || exit 1
echo TEACHER_JOB_DONE
