"""GPU smoke test: JAX-CUDA + Torch-CUDA + OSMesa rendering in ONE process."""
import os, sys, time
import numpy as np

import jax
print("jax backend:", jax.default_backend(), "devices:", jax.devices())
assert jax.default_backend() == "gpu", "JAX did not see the GPU"

import torch
print("torch cuda:", torch.cuda.is_available(), torch.cuda.device_count(),
      torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), "torch did not see the GPU"
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
print("torch matmul ok, mean=", float((x @ x).mean()))

from hw2 import envs as E
E.use_osmesa()                      # before any mujoco import

env = E.make_env(impl="jax")
spec = E.describe_env(env)
print("SPEC:", spec.as_dict())

import jax.numpy as jp
reset, step = jax.jit(env.reset), jax.jit(env.step)
st = reset(jax.random.PRNGKey(0))
print("obs:", st.obs.shape, "act:", env.action_size,
      "info:", list(st.info.keys()), "metrics:", list(st.metrics.keys()))

lo, hi = spec.target_rel_slice
traj = [st]
for i in range(40):
    st = step(st, jp.zeros(env.action_size))
    traj.append(st)
    d_obs = float(jp.linalg.norm(st.obs[lo:hi]))
    d_dat = float(E.box_target_distance(env, st))
    assert abs(d_obs - d_dat) < 1e-4, f"obs slice mismatch {d_obs} vs {d_dat}"
print(f"obs/target-slice consistency OK (final dist {d_dat:.4f} m)")

B = 512
vreset, vstep = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
s = vreset(jax.random.split(jax.random.PRNGKey(1), B))
acts = jp.zeros((B, env.action_size))
s = vstep(s, acts); jax.block_until_ready(s.obs)
t0 = time.time()
for _ in range(50):
    s = vstep(s, acts)
jax.block_until_ready(s.obs)
print(f"vmapped throughput: B={B} {50*B/(time.time()-t0):,.0f} env-steps/s")

t0 = time.time()
frames = env.render(traj[::5], height=240, width=320)
print(f"rendered {len(frames)} frames in {time.time()-t0:.1f}s shape={np.asarray(frames[0]).shape}")
import imageio.v3 as iio
out = os.path.join(os.environ["PROJ"], "results", "smoke_frame.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
iio.imwrite(out, np.asarray(frames[0]))
print("wrote", out)
print("SMOKE_OK")
