#!/bin/bash
# GPU job: re-score saved rollouts, failure analysis + verified videos.
set -uo pipefail
J="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$J/../env.sh"; cd "$PROJ"
"$PY" scripts/rescore_episodes.py
rm -f videos/student_*.mp4 videos/student_*_frames.npz
"$PY" scripts/stage2b_failure.py 2>&1 | grep -v "Device or resource busy"
