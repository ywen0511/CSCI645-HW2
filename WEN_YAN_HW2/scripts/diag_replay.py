"""Is the D1 replay deviation chaotic amplification or an obs/action misalignment?

If the (obs, action) pairing or the seeding were wrong, the deviation would be
large at the FIRST step. If the simulator is merely not bit-identical between a
vmapped batch of 512 and a single-env replay, the deviation starts at ~1e-7 and
grows, which is ordinary chaos in a contact-rich system and is harmless.
"""
import numpy as np

import common as CM
from hw2 import envs as E
E.configure_rendering()
import jax, jax.numpy as jp
from hw2.data import DemoSet

env = E.make_env("PandaPickCube", impl="jax")
spec = E.describe_env(env)
d = DemoSet.load(str(CM.DATA / "demos_sigma0.npz"))
idxs = d.meta["subset_episode_indices"]
keys = jax.random.split(jax.random.PRNGKey(d.meta["collect_key_base"]), 512)

reset, step = jax.jit(env.reset), jax.jit(env.step)
vreset = jax.jit(jax.vmap(env.reset))
out = {}

# Does a single reset match a vmapped reset bit-for-bit?
j = 0
k = keys[int(idxs[j])]
o_single = np.asarray(reset(k).obs)
o_batch = np.asarray(vreset(keys).obs)[int(idxs[j])]
out["reset_single_vs_vmap_maxdev"] = float(np.abs(o_single - o_batch).max())
out["reset_vs_stored_maxdev"] = float(np.abs(o_single - d.obs[d.episode_slice(j)][0]).max())
print(f"[reset] single vs vmapped reset: {out['reset_single_vs_vmap_maxdev']:.3e}")
print(f"[reset] single reset vs stored obs[0]: {out['reset_vs_stored_maxdev']:.3e}")

# Per-timestep divergence while replaying the stored actions
curves = []
for j in range(3):
    sl = d.episode_slice(j)
    acts = d.actions[sl]
    st = reset(keys[int(idxs[j])])
    devs = [float(np.abs(np.asarray(st.obs) - d.obs[sl][0]).max())]
    for t in range(acts.shape[0] - 1):
        st = step(st, jp.asarray(acts[t]))
        devs.append(float(np.abs(np.asarray(st.obs) - d.obs[sl][t + 1]).max()))
    curves.append(devs)
    first_big = next((i for i, v in enumerate(devs) if v > 1e-3), None)
    print(f"[ep{j}] dev at t=0..6: {[f'{v:.2e}' for v in devs[:7]]}")
    print(f"       first t with dev>1e-3: {first_big}   final dev: {devs[-1]:.3e}")
out["per_step_deviation"] = curves
out["first_exceed_1e3"] = [next((i for i, v in enumerate(c) if v > 1e-3), None)
                           for c in curves]
CM.write_json(CM.RESULTS / "diag" / "diag_replay.json", out)
verdict = ("CHAOS (benign): deviation starts at numerical noise and grows"
           if all((c[0] < 1e-4 and c[1] < 1e-3) for c in curves)
           else "MISALIGNMENT (bug): deviation is large immediately")
print("VERDICT:", verdict)
print("REPLAY_DIAG_DONE")
