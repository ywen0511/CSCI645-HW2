#!/bin/bash
# Submit a self-terminating GPU job to gc216 only.
#   submit.sh <jobname> <ngpu> <wall_minutes> <slots> <command ...>
# The workload IS the job, so the reservation ends exactly when compute ends.
# NOTE: profile.lsf ends with `return 1` on this cluster, so it must be sourced
# with errexit disabled or the script dies before reaching bsub.
set -uo pipefail
set +e
. /lsf/nahpc_wrn/conf/profile.lsf >/dev/null 2>&1
set -e
NAME=$1; NGPU=$2; WALL=$3; SLOTS=$4; shift 4
PROJ=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci645-hw2
mkdir -p "$PROJ/logs"
bsub -q gpu_large -m dcwipphhgc216 \
     -gpu "num=${NGPU}:mode=exclusive_process" \
     -n "$SLOTS" -R "rusage[mem=48000]" -W "$WALL" \
     -J "$NAME" -o "$PROJ/logs/${NAME}.%J.log" \
     "$@"
