"""Throughput benchmark: which MJX backend/batch size makes PPO affordable?"""
import os, time, sys
import numpy as np
import jax, jax.numpy as jp

print("jax", jax.__version__, jax.devices())
try:
    import mujoco_warp; print("mujoco_warp AVAILABLE", getattr(mujoco_warp,"__version__","?"))
except Exception as e:
    print("mujoco_warp MISSING:", type(e).__name__, e)

sys.path.insert(0, os.path.join(os.environ["PROJ"], "code"))
from hw2 import envs as E

def bench(impl, B, n=30):
    try:
        env = E.make_env(impl=impl)
    except Exception as e:
        return f"make_env FAIL: {type(e).__name__}: {str(e)[:120]}"
    try:
        vreset, vstep = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
        s = vreset(jax.random.split(jax.random.PRNGKey(0), B))
        a = jp.zeros((B, env.action_size))
        t0=time.time(); s = vstep(s, a); jax.block_until_ready(s.obs)
        compile_s = time.time()-t0
        t0=time.time()
        for _ in range(n): s = vstep(s, a)
        jax.block_until_ready(s.obs)
        dt=time.time()-t0
        return f"{n*B/dt:>12,.0f} steps/s   (compile {compile_s:5.1f}s, {dt/n*1e3:6.1f} ms/iter)"
    except Exception as e:
        return f"step FAIL: {type(e).__name__}: {str(e)[:120]}"

for impl in ["jax", "warp"]:
    for B in [512, 2048, 4096]:
        print(f"impl={impl:5s} B={B:5d}  {bench(impl,B)}", flush=True)
print("BENCH_DONE")
