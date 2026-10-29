#!/bin/bash
# OPTIONAL helper for an LSF cluster (how the reported runs were launched).
#   submit.sh <jobname> <ngpu> <wall_minutes> <slots> <command ...>
# The workload IS the job, so the reservation ends when the compute ends.
# Configure with HW2_LSF_QUEUE (default gpu_large) and HW2_LSF_HOST (optional).
# Nothing else in the pipeline needs LSF: run the same commands directly.
set -uo pipefail
[ -n "${LSF_PROFILE:-}" ] && { set +e; . "$LSF_PROFILE" >/dev/null 2>&1; }
set -e
NAME=$1; NGPU=$2; WALL=$3; SLOTS=$4; shift 4
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
mkdir -p "$HW2_RUNS/logs"
HOSTARG=(); [ -n "${HW2_LSF_HOST:-}" ] && HOSTARG=(-m "$HW2_LSF_HOST")
GPUARG=(); [ "$NGPU" -gt 0 ] && GPUARG=(-gpu "num=${NGPU}:mode=shared:j_exclusive=yes")
bsub -q "${HW2_LSF_QUEUE:-gpu_large}" "${HOSTARG[@]}" "${GPUARG[@]}" \
     -n "$SLOTS" -R "rusage[mem=48000] span[hosts=1]" -W "$WALL" \
     -J "$NAME" -o "$HW2_RUNS/logs/${NAME}.%J.log" "$@"
