"""PandaPickCube environment helpers: construction, success criterion, rendering.

Two facts about the upstream environment drive this file
(mujoco_playground @ d59156e, manipulation/franka_emika_panda/pick.py):

1. `done` is raised ONLY on out-of-bounds / NaN, never on task completion, so
   every episode runs the full `episode_length` steps unless the box leaves the
   workspace. There is therefore no early-termination signal to learn from.
2. There is NO success field anywhere in `state.info` or `state.metrics`. The
   only task-progress flag is `info["reached_box"]` (gripper within 12 mm of the
   box). Task success must be defined by us; we define it geometrically from the
   box->target distance and report sensitivity to the threshold.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Warren GPU nodes ship no NVIDIA graphics userspace (no libEGL_nvidia), so the
# upstream `MUJOCO_GL=egl` recipe cannot work here. Rendering must go through a
# software OSMesa stack, which needs no GPU at all.
OSMESA_PREFIX = "/data/rtxw01/nbak/robot_intel/kztrgg/envs/osmesa_hw2"


def use_osmesa(prefix: str = OSMESA_PREFIX) -> None:
    """Configure software GL. MUST be called BEFORE `import mujoco`.

    `LD_LIBRARY_PATH` alone is not sufficient: PyOpenGL resolves the library
    through `ctypes.util.find_library("OSMesa")`, which on Linux consults
    `ldconfig`/`ld` and NOT `LD_LIBRARY_PATH`. When that returns None, PyOpenGL
    silently loads its GL module as None and the first attribute access dies
    with `AttributeError: 'NoneType' object has no attribute 'glGetError'` --
    and because mujoco >= 3.15 imports its renderer at package-import time,
    even `import mujoco` fails. So we resolve the path ourselves, preload it
    with RTLD_GLOBAL, and patch `find_library` to answer for the GL names.
    """
    import ctypes
    import ctypes.util

    lib = os.path.join(prefix, "lib")
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if lib not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{cur}:{lib}" if cur else lib

    def _resolve(name: str):
        for cand in (f"lib{name}.so.8", f"lib{name}.so.1", f"lib{name}.so.0",
                     f"lib{name}.so"):
            p = os.path.join(lib, cand)
            if os.path.exists(p):
                return p
        return None

    osmesa = _resolve("OSMesa")
    if osmesa is None:
        raise RuntimeError(
            f"libOSMesa not found under {lib}. Rebuild with: "
            f"conda create -p {prefix} -c conda-forge mesalib=23.0.1"
        )
    # Preload so that dependent symbols resolve inside this process.
    for dep in ("glapi", "OSMesa"):
        p = _resolve(dep)
        if p:
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)

    _orig = ctypes.util.find_library

    def _patched(name):
        if name in ("OSMesa", "glapi", "GL", "GLU"):
            r = _resolve(name)
            if r:
                return r
        return _orig(name)

    ctypes.util.find_library = _patched


TASK = "PandaPickCube"

# Success: the box must come within SUCCESS_RADIUS of the target and STAY there
# for SUCCESS_HOLD consecutive control steps, so that a fast fly-through of the
# target region does not count as a pick-and-place.
SUCCESS_RADIUS = 0.05   # metres
SUCCESS_HOLD = 10       # control steps (= 0.2 s at ctrl_dt = 0.02)


def make_env(task: str = TASK, impl: str = "jax",
             config_overrides: Optional[Dict[str, Any]] = None):
    """Build a playground env on the JAX backend.

    The upstream default is `impl="warp"`; we pin `impl="jax"` to match the
    homework's reference snippet and to keep everything inside one JAX runtime.
    """
    from mujoco_playground import registry

    overrides = {"impl": impl}
    if config_overrides:
        overrides.update(config_overrides)
    return registry.load(task, config_overrides=overrides)


@dataclass
class EnvSpec:
    obs_dim: int
    act_dim: int
    episode_length: int
    ctrl_dt: float
    nq: int
    nv: int
    target_rel_slice: Tuple[int, int]  # obs slice holding target_pos - box_pos
    gripper_slice: Tuple[int, int] = (0, 0)   # absolute gripper site position
    box_rel_slice: Tuple[int, int] = (0, 0)   # box_pos - gripper_pos

    def as_dict(self) -> dict:
        return {
            "obs_dim": self.obs_dim, "act_dim": self.act_dim,
            "episode_length": self.episode_length, "ctrl_dt": self.ctrl_dt,
            "nq": self.nq, "nv": self.nv,
            "target_rel_slice": list(self.target_rel_slice),
            "gripper_slice": list(self.gripper_slice),
            "box_rel_slice": list(self.box_rel_slice),
        }


def describe_env(env) -> EnvSpec:
    """Derive dimensions and locate `target_pos - box_pos` inside the obs vector.

    Obs layout in pick.py `_get_obs` is
        qpos(nq) | qvel(nv) | gripper_pos(3) | gripper_mat[3:](6) |
        box_mat[3:](6) | box_pos-gripper_pos(3) | target_pos-box_pos(3) | ...
    so the target-relative block starts at nq+nv+3+6+6+3.
    """
    nq = int(env.mjx_model.nq)
    nv = int(env.mjx_model.nv)
    start = nq + nv + 3 + 6 + 6 + 3
    obs_size = env.observation_size
    obs_dim = int(obs_size) if isinstance(obs_size, int) else int(obs_size["state"][0])
    g0 = nq + nv                 # gripper_pos
    b0 = nq + nv + 3 + 6 + 6     # box_pos - gripper_pos
    return EnvSpec(
        obs_dim=obs_dim,
        act_dim=int(env.action_size),
        episode_length=int(env._config.episode_length),
        ctrl_dt=float(env._config.ctrl_dt),
        nq=nq, nv=nv,
        target_rel_slice=(start, start + 3),
        gripper_slice=(g0, g0 + 3),
        box_rel_slice=(b0, b0 + 3),
    )


def decode_positions(obs, spec: "EnvSpec"):
    """Recover absolute gripper / box / target positions from the obs vector.

    `_get_obs` stores the gripper site position absolutely and the box and
    target only relatively, so the absolute positions are reconstructed by
    chaining the offsets:
        box    = gripper + (box - gripper)
        target = box     + (target - box)
    """
    import numpy as _np
    o = _np.asarray(obs)
    g = o[..., spec.gripper_slice[0]:spec.gripper_slice[1]]
    box = g + o[..., spec.box_rel_slice[0]:spec.box_rel_slice[1]]
    tgt = box + o[..., spec.target_rel_slice[0]:spec.target_rel_slice[1]]
    return {"gripper": g, "box": box, "target": tgt}


def box_target_distance(env, state) -> Any:
    """Box->target distance from simulator state (ground truth, not the obs)."""
    import jax.numpy as jp
    box_pos = state.data.xpos[env._obj_body]
    return jp.linalg.norm(state.info["target_pos"] - box_pos)


def success_from_distances(dists: np.ndarray,
                           radius: float = SUCCESS_RADIUS,
                           hold: int = SUCCESS_HOLD) -> np.ndarray:
    """Per-episode success from a (n_episodes, T) array of box->target distances.

    Success == `hold` consecutive steps with distance < `radius`.
    """
    dists = np.asarray(dists)
    if dists.ndim == 1:
        dists = dists[None, :]
    inside = (dists < radius).astype(np.int32)
    n, T = inside.shape
    if hold <= 1:
        return inside.any(1)
    # Longest run of 1s per row >= hold, via cumulative-sum over a sliding window.
    csum = np.concatenate([np.zeros((n, 1), np.int32), np.cumsum(inside, 1)], 1)
    if T < hold:
        return np.zeros((n,), dtype=bool)
    win = csum[:, hold:] - csum[:, :-hold]      # (n, T-hold+1) sums of each window
    return (win >= hold).any(1)


# --------------------------------------------------------------------------
# rendering (OSMesa, CPU only)
# --------------------------------------------------------------------------
def render_trajectory(env, states, height: int = 480, width: int = 640,
                      stride: int = 2):
    """Render a list of env States to a list of RGB frames."""
    return env.render(states[::stride], height=height, width=width)


def ensure_ffmpeg():
    """Make an ffmpeg binary available without root.

    These nodes have no system ffmpeg (`mediapy` dies with "Program 'ffmpeg' is
    not found"), but the pip package `imageio-ffmpeg` ships a static build. Its
    binary is named `ffmpeg-linux-x86_64-vX.Y.Z`, so simply putting its folder
    on PATH is not enough -- we symlink it to the plain name `ffmpeg` in a
    project-local bin directory and prepend that.
    """
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        src = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    bindir = os.path.join(os.environ.get("PROJ", os.path.expanduser("~")), ".bin")
    os.makedirs(bindir, exist_ok=True)
    link = os.path.join(bindir, "ffmpeg")
    if not os.path.exists(link):
        try:
            os.symlink(src, link)
        except FileExistsError:
            pass
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", src)
    return link


def write_video(path: str, frames, fps: float) -> str:
    """Write an mp4; fall back to GIF if no encoder can be found."""
    import numpy as _np
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = [_np.asarray(f) for f in frames]
    ensure_ffmpeg()
    try:
        import imageio.v2 as iio2
        iio2.mimwrite(path, arr, fps=float(fps), codec="libx264",
                      macro_block_size=1, quality=8)
        return path
    except Exception as e:                      # pragma: no cover
        print(f"[video] mp4 encode failed ({type(e).__name__}: {e}); writing GIF")
        gif = os.path.splitext(path)[0] + ".gif"
        import imageio.v2 as iio2
        iio2.mimwrite(gif, arr, duration=1.0 / float(fps), loop=0)
        return gif


def write_frames_npz(path: str, frames, n_keep: int = 8) -> str:
    """Save uniformly sampled frames as an array.

    The film-strip figure reads these instead of decoding the mp4, so the figure
    does not depend on a video decoder being installed (pyav is not).
    """
    import numpy as _np
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = _np.stack([_np.asarray(f) for f in frames])
    idx = _np.linspace(0, len(arr) - 1, min(n_keep, len(arr))).astype(int)
    _np.savez_compressed(path, frames=arr[idx], frame_index=idx,
                         n_total=len(arr))
    return path
