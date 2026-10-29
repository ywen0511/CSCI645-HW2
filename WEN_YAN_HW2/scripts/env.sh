# Shared environment for all stages. Source it; do not execute it.
#   . scripts/env.sh
# Everything is derived from this checkout. Optional overrides:
#   HW2_PYTHON         python of the environment from scripts/setup_env.sh
#   HW2_RUNS           checkpoints / large run outputs (default: $PROJ/runs)
#   HW2_GL, HW2_OSMESA_PREFIX   video rendering backend (see code/hw2/envs.py)
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJ="${PROJ:-$(cd "$_here/.." && pwd)}"
export PY="${HW2_PYTHON:-python}"
export HW2_RUNS="${HW2_RUNS:-$PROJ/runs}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/hw2_mpl_$(id -u)}"
mkdir -p "$HW2_RUNS" "$MPLCONFIGDIR"
# JAX must not grab the whole GPU: torch shares the device during evaluation.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# full-fp32 matmuls on Ampere, per the MuJoCo Playground reproducibility note
export JAX_DEFAULT_MATMUL_PRECISION=highest
# one torch-inductor compile worker per process (nothing here uses compile)
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="$PROJ/code:$PROJ/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
unset _here
