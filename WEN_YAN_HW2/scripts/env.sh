# Shared environment for all CSCI645-HW2 stages. Source, do not execute.
export BASE=/data/rtxw01/nbak/robot_intel/kztrgg
export PROJ=$BASE/projs/code/csci645-hw2
export PY=$BASE/envs/hw2/bin/python
export HF_HOME=$BASE/hf_cache
export TMPDIR=$BASE/tmp
export MPLCONFIGDIR=$BASE/tmp/mpl
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

# JAX must not grab the whole GPU: torch shares the device during evaluation.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Ampere full-fp32 matmuls, per the MuJoCo Playground reproducibility note.
export JAX_DEFAULT_MATMUL_PRECISION=highest

# NOTE: MUJOCO_GL / PYOPENGL_PLATFORM / the OSMesa lib path are deliberately NOT
# exported here. These nodes have no NVIDIA graphics userspace, so rendering goes
# through software OSMesa, and that has to be configured from inside Python by
# hw2.envs.use_osmesa() BEFORE mujoco is imported (it patches the ctypes loader).
# Exporting MUJOCO_GL=osmesa here would make a bare `import mujoco` fail in any
# stage that had not called use_osmesa() first.

export PYTHONPATH=$PROJ/code:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
