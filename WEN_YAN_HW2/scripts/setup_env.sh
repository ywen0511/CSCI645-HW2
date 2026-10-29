#!/bin/bash
# Create the Python environment used for every stage of this submission.
#
#   ENVP=/path/to/new/env [OSMESA_PREFIX=/path/to/osmesa] [CONDA=conda] \
#       bash scripts/setup_env.sh
#
# ENVP          conda prefix to create (required)
# OSMESA_PREFIX optional: also build a software-GL (OSMesa) prefix for video
#               rendering on machines without EGL/GLX. Skip it on a desktop
#               with a working GPU driver and use HW2_GL=egl or glfw instead.
# CONDA         conda executable (default: `conda` on PATH)
#
# The pins below are the versions the reported results were produced with
# (see results/environment.json and configs/requirements.lock.txt).
set -euo pipefail
: "${ENVP:?set ENVP to the conda prefix to create}"
CONDA=${CONDA:-conda}
PLAYGROUND_COMMIT=d59156e099a8785dd58fbce2e25a6ee1f7b5800c

if [ ! -x "$ENVP/bin/python" ]; then
  "$CONDA" create -y -p "$ENVP" python=3.12
fi
PIP="$ENVP/bin/python -m pip"
$PIP install --upgrade pip
# jax/flax MUST be pinned before brax: brax 0.14.2 still calls
# jax.device_put_replicated, which jax >= 0.10 removed.
$PIP install "jax[cuda12]==0.9.2" "jaxlib==0.9.2" "flax==0.11.2"
$PIP install "mujoco==3.15.0" "mujoco-mjx==3.15.0"
$PIP install "git+https://github.com/google-deepmind/mujoco_playground@${PLAYGROUND_COMMIT}"
$PIP install "brax==0.14.2" "jax[cuda12]==0.9.2" "jaxlib==0.9.2" "flax==0.11.2"
$PIP install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
$PIP install mediapy "imageio[ffmpeg]" matplotlib pandas scipy tqdm einops pytest pypdf

"$ENVP/bin/python" - <<'EOF'
import jax, brax, mujoco, torch
assert hasattr(jax, "device_put_replicated"), "jax too new for brax 0.14.2"
print("jax", jax.__version__, "brax", brax.__version__, "mujoco", mujoco.__version__,
      "torch", torch.__version__)
EOF

if [ -n "${OSMESA_PREFIX:-}" ] && [ ! -e "$OSMESA_PREFIX/lib/libOSMesa.so.8" ]; then
  "$CONDA" create -y -p "$OSMESA_PREFIX" -c conda-forge mesalib=23.0.1
fi
echo SETUP_ENV_DONE
