"""Stage 1: train the PPO teacher, evaluate it, and collect all demo datasets.

Teacher training and demonstration collection live in ONE job so that the
policy used for collection is the exact in-memory object returned by training
(no re-deriving an inference function from a pickle, which is where observation
normalisation silently goes wrong).
"""
import argparse, functools, json, os, pickle, time
import numpy as np


def _jsonable(o):
    """numpy scalars/arrays are not JSON-serialisable; convert on the way out."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")

import sys
sys.path.insert(0, os.path.join(os.environ["PROJ"], "code"))

from hw2 import envs as E
E.use_osmesa()                      # before any mujoco import

import jax
import jax.numpy as jp
from hw2 import collect as C
from hw2.evaluate import summarize, wilson_ci

TASK = "PandaPickCube"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_timesteps", type=int, default=None)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--n_demo_target", type=int, default=100)
    ap.add_argument("--collect_batch", type=int, default=256)
    ap.add_argument("--noise_levels", default="0.0,0.1,0.3")
    a = ap.parse_args()

    PROJ = os.environ["PROJ"]
    out = a.out or os.path.join(PROJ, "runs", "teacher")
    os.makedirs(out, exist_ok=True)
    data_dir = os.path.join(PROJ, "data")
    os.makedirs(data_dir, exist_ok=True)

    # ---------------- 1. train the teacher ----------------
    from hw2.teacher import train_teacher
    import functools as ft
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from brax.training.acme import running_statistics
    from brax.training import types as brax_types
    from mujoco_playground import wrapper
    from hw2.teacher import ppo_config

    env = E.make_env(TASK, impl="jax")
    eval_env = E.make_env(TASK, impl="jax")
    spec = E.describe_env(env)
    print("SPEC:", spec.as_dict(), flush=True)

    cfg = dict(ppo_config(TASK))
    training_params = dict(cfg)
    nf = training_params.pop("network_factory", None)
    network_factory = (ft.partial(ppo_networks.make_ppo_networks, **nf) if nf
                       else ppo_networks.make_ppo_networks)
    if a.num_timesteps:
        training_params["num_timesteps"] = int(a.num_timesteps)
    print("[teacher] training params:", json.dumps(
        {k: str(v) for k, v in training_params.items()}, indent=2), flush=True)

    progress, t0 = [], time.time()
    def progress_fn(step, metrics):
        row = {"step": int(step), "wall_s": time.time() - t0}
        for k, v in metrics.items():
            try: row[k] = float(v)
            except Exception: pass
        progress.append(row)
        print(f"[teacher] step={step:>10,} reward="
              f"{row.get('eval/episode_reward', float('nan')):.2f} "
              f"({row['wall_s']:.0f}s)", flush=True)
        json.dump(progress, open(os.path.join(out, "ppo_progress.json"), "w"), indent=2, default=_jsonable)

    ckpt_dir = os.path.join(out, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
    snaps = []
    def policy_params_fn(current_step, make_policy, params):
        del make_policy
        p = os.path.join(ckpt_dir, f"params_step{int(current_step):010d}.pkl")
        pickle.dump(jax.device_get(params), open(p, "wb"))
        snaps.append(int(current_step))

    make_inference_fn, params, _ = ft.partial(
        ppo.train, **training_params, network_factory=network_factory,
        seed=a.seed, wrap_env_fn=wrapper.wrap_for_brax_training,
    )(environment=env, eval_env=eval_env, progress_fn=progress_fn,
      policy_params_fn=policy_params_fn)

    pickle.dump(jax.device_get(params), open(os.path.join(out, "teacher_params.pkl"), "wb"))
    teacher_wall = time.time() - t0
    print(f"[teacher] trained in {teacher_wall:.0f}s", flush=True)

    # deterministic teacher policy, used for every rollout below
    policy_fn = jax.jit(make_inference_fn(params, deterministic=True))

    radius = E.SUCCESS_RADIUS; hold = E.SUCCESS_HOLD
    report = {"task": TASK, "spec": spec.as_dict(), "seed": a.seed,
              "teacher_wall_s": teacher_wall,
              "ppo_params": {k: str(v) for k, v in training_params.items()},
              "success_radius": radius, "success_hold": hold,
              "snapshots": sorted(snaps)}

    # ---------------- 2. teacher evaluation on the HELD-OUT seeds ----------------
    for sigma in [float(x) for x in a.noise_levels.split(",")]:
        res = C.rollout(env, policy_fn, C.eval_seeds(a.n_eval),
                        spec.episode_length, action_noise=sigma,
                        target_rel_slice=tuple(spec.target_rel_slice),
                        box_rel_slice=tuple(spec.box_rel_slice))
        s = summarize({"dist": res["dist"], "rewards": res["rewards"],
                       "done": res["done"], "grip_dist": res["grip_dist"]},
                      radius, hold)
        s.pop("per_episode_success", None)
        # Threshold sweep: tells us whether a low score means "never lifted the
        # box" or "lifted it to just outside the tolerance".
        md = res["dist"].min(1)
        s["success_by_radius"] = {
            f"{r:g}": float(np.mean(E.success_from_distances(res["dist"], r, hold)))
            for r in (0.03, 0.05, 0.08, 0.10, 0.15)}
        s["min_dist_percentiles"] = {
            f"p{q}": float(np.percentile(md, q)) for q in (10, 25, 50, 75, 90)}
        report[f"teacher_eval_sigma{sigma}"] = s
        print(f"[teacher-eval] sigma={sigma} success={s['success_rate']:.3f} "
              f"CI={s['success_ci95']} return={s['return_mean']:.1f} "
              f"reached_box={s['reached_box_frac']} "
              f"min_dist_med={s['min_dist_percentiles']['p50']:.4f}", flush=True)
        print(f"               success_by_radius={s['success_by_radius']}", flush=True)

    # ---------------- 3. collect demonstration datasets ----------------
    datasets = {}
    for sigma in [float(x) for x in a.noise_levels.split(",")]:
        tag = f"sigma{sigma:g}"
        collected, attempts, base = [], 0, 1000 + int(sigma * 1000)
        # Clean set: keep collecting until we have enough SUCCESSFUL episodes.
        # Noisy sets: keep a fixed number of episodes regardless of outcome,
        # because degraded data quality is the independent variable.
        need_success = (sigma == 0.0)
        pool = None
        while True:
            keys = jax.random.split(jax.random.PRNGKey(base + attempts), a.collect_batch)
            traj = C.rollout(env, policy_fn, keys, spec.episode_length,
                             action_noise=sigma,
                             target_rel_slice=tuple(spec.target_rel_slice),
                             box_rel_slice=tuple(spec.box_rel_slice))
            ds = C.to_demoset(traj, radius, hold,
                              meta={"name": tag, "action_noise": sigma,
                                    "collect_key_base": base + attempts})
            pool = ds if pool is None else _concat(pool, ds)
            attempts += 1
            have = int(pool.success.sum()) if need_success else pool.n_episodes
            print(f"[collect] sigma={sigma} batch{attempts} pool={pool.n_episodes} "
                  f"succ={int(pool.success.sum())} ({pool.success.mean():.3f})", flush=True)
            if have >= a.n_demo_target or attempts >= 6:
                break
        if need_success:
            keep = np.flatnonzero(pool.success)[: a.n_demo_target]
            if keep.size < a.n_demo_target:
                print(f"[collect] *** WARNING sigma={sigma}: only {keep.size} "
                      f"successful episodes out of {pool.n_episodes} "
                      f"(target {a.n_demo_target}). The teacher is too weak; "
                      f"saving the raw pool for inspection and continuing.",
                      flush=True)
                raw = os.path.join(data_dir, f"rawpool_{tag}.npz")
                pool.save(raw)
                report.setdefault("warnings", []).append({
                    "stage": "collect", "sigma": sigma,
                    "n_success": int(keep.size), "n_pool": int(pool.n_episodes),
                    "raw_pool": raw})
                if keep.size == 0:
                    continue
        else:
            keep = np.arange(min(a.n_demo_target, pool.n_episodes))
        final = pool.subset(keep.tolist())
        final.meta.update({"name": tag, "action_noise": sigma,
                           "filtered_to_success": bool(need_success),
                           "raw_pool_episodes": int(pool.n_episodes),
                           "raw_pool_success_rate": float(pool.success.mean())})
        path = os.path.join(data_dir, f"demos_{tag}.npz")
        final.save(path)
        datasets[tag] = {"path": path, "summary": final.summary(),
                         "meta": final.meta}
        print(f"[collect] sigma={sigma} saved {final.n_episodes} eps -> {path}", flush=True)
        print(f"          {json.dumps(final.summary())}", flush=True)
    report["datasets"] = datasets

    # ---------------- 4. expert demonstration videos ----------------
    vid_dir = os.path.join(PROJ, "videos"); os.makedirs(vid_dir, exist_ok=True)
    from hw2.data import DemoSet
    clean = DemoSet.load(datasets["sigma0"]["path"])
    made = []
    # Re-roll two demo episodes keeping States so we can render them.
    dkeys = C.demo_seeds(a.collect_batch, base=1000)
    for i in range(min(2, a.collect_batch)):
        states = C.rollout_states(env, policy_fn, dkeys[i], spec.episode_length)
        d = np.array([float(jp.linalg.norm(s.obs[spec.target_rel_slice[0]:spec.target_rel_slice[1]]))
                      for s in states])
        ok = bool(E.success_from_distances(d[None], radius, hold)[0])
        frames = env.render(states[::2], height=360, width=480)
        p = os.path.join(vid_dir, f"expert_demo{i}_{'success' if ok else 'failure'}.mp4")
        written = E.write_video(p, frames, fps=round(1.0 / spec.ctrl_dt / 2))
        npz = os.path.join(vid_dir, f"expert_demo{i}_frames.npz")
        E.write_frames_npz(npz, frames, n_keep=8)
        made.append({"path": written, "frames_npz": npz, "success": ok,
                     "min_dist": float(d.min())})
        print(f"[video] {p} success={ok} min_dist={d.min():.4f}", flush=True)
    report["expert_videos"] = made

    json.dump(report, open(os.path.join(out, "stage1_report.json"), "w"), indent=2, default=_jsonable)
    print("STAGE1_DONE", flush=True)


def _concat(a, b):
    from hw2.data import DemoSet
    off = a.obs.shape[0]
    return DemoSet(
        obs=np.concatenate([a.obs, b.obs], 0),
        actions=np.concatenate([a.actions, b.actions], 0),
        rewards=np.concatenate([a.rewards, b.rewards], 0),
        episode_ends=np.concatenate([a.episode_ends, b.episode_ends + off]),
        success=np.concatenate([a.success, b.success]),
        meta=a.meta,
    )


if __name__ == "__main__":
    main()
