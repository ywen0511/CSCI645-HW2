"""Stage 1R: rebuild the PPO teacher, evaluate it (protocol v2), audit datasets.

The historical teacher checkpoint (teacher_params.pkl) was not recoverable, so
the teacher is retrained with the historical configuration (MuJoCo Playground
`brax_ppo_config("PandaPickCube")`, num_timesteps = 60M, seed 0). The
RETRAINED teacher is a NEW artifact: it did not generate the committed
demonstration datasets, and nothing here re-collects them. This stage

  1. trains the teacher (skipped if runs/teacher holds a matching one),
  2. checks that the policy rebuilt from the pickle reproduces the in-memory
     policy (observation normalisation is where this silently breaks),
  3. evaluates it on the original 100 held-out initial conditions at every
     action-noise level used in the study, with the termination-aware metrics,
  4. audits each committed dataset: initial states vs. the recorded collection
     keys, success labels under protocol v2, and how closely the new teacher's
     actions match the stored actions (i.e. whether it behaves like the
     teacher that generated the data).

    python scripts/stage1r_teacher.py [--num_timesteps 60000000]
"""
import argparse
import functools as ft
import json
import os
import pickle
import time

import numpy as np

import common as CM
from hw2 import envs as E
E.configure_rendering()

SIGMAS = (0.0, 0.1, 0.3, 0.5, 1.0, 2.0)
DATASETS = {0.0: "demos_sigma0.npz", 0.1: "demos_sigma0.1.npz",
            0.3: "demos_sigma0.3.npz", 0.5: "demos_sigma0.5.npz",
            1.0: "demos_sigma1.npz", 2.0: "demos_sigma2.npz"}


def train_teacher(out, seed, num_timesteps, env, eval_env):
    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper
    from hw2.teacher import ppo_config

    tp = dict(ppo_config(E.TASK))
    nf = tp.pop("network_factory", None)
    netf = (ft.partial(ppo_networks.make_ppo_networks, **nf) if nf
            else ppo_networks.make_ppo_networks)
    tp["num_timesteps"] = int(num_timesteps)
    progress, t0 = [], time.time()

    def progress_fn(step, metrics):
        row = {"step": int(step), "wall_s": time.time() - t0}
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except Exception:
                pass
        progress.append(row)
        print(f"[teacher] step={step:>11,} reward="
              f"{row.get('eval/episode_reward', float('nan')):.1f} "
              f"({row['wall_s']:.0f}s)", flush=True)
        CM.write_json(out / "ppo_progress.json", progress)

    make_inf, params, _ = ft.partial(
        ppo.train, **tp, network_factory=netf, seed=seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )(environment=env, eval_env=eval_env, progress_fn=progress_fn)
    with open(out / "teacher_params.pkl", "wb") as f:
        pickle.dump(jax.device_get(params), f)
    info = {"seed": seed, "num_timesteps_requested": int(num_timesteps),
            "num_timesteps_reached": progress[-1]["step"] if progress else None,
            "ppo_params": {k: str(v) for k, v in tp.items()},
            "wall_s": time.time() - t0,
            "provenance": "retrained 2026-10; NOT the policy that generated data/*.npz"}
    return jax.jit(make_inf(params, deterministic=True)), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_timesteps", type=int, default=60_000_000)
    ap.add_argument("--n_eval", type=int, default=100)
    a = ap.parse_args()

    import jax
    import jax.numpy as jp
    from hw2 import collect as C
    from hw2 import metrics as MT
    from hw2.data import DemoSet
    from hw2.evaluate import key_data, save_episodes, score, summarize
    from hw2.teacher import load_teacher_policy

    out = CM.RUNS / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    env = E.make_env(E.TASK, impl="jax")
    spec = E.describe_env(env)
    radius, hold = E.SUCCESS_RADIUS, E.SUCCESS_HOLD
    pkl = out / "teacher_params.pkl"
    ident = {"seed": a.seed, "num_timesteps": a.num_timesteps,
             "playground": "d59156e099a8785dd58fbce2e25a6ee1f7b5800c"}
    report_path = CM.RESULTS / "teacher" / "teacher_report.json"
    report = {}

    # ---------------- 1-2. train (or reuse) and reload check ----------------
    info_path = out / "teacher_info.json"
    if pkl.exists() and info_path.exists() and CM.read_json(info_path).get("identity") == ident:
        info = CM.read_json(info_path)
        print("[teacher] reusing", CM.rel(pkl), flush=True)
        policy_mem = None
    else:
        policy_mem, info = train_teacher(out, a.seed, a.num_timesteps, env,
                                         E.make_env(E.TASK, impl="jax"))
        info["identity"] = ident
    info["params_sha256"] = CM.sha256_file(pkl)
    _, policy = load_teacher_policy(E.TASK, str(pkl))
    keys = C.eval_seeds(a.n_eval)
    if policy_mem is not None:
        o = jax.vmap(env.reset)(keys).obs
        dev = float(jp.abs(policy_mem(o, jax.random.PRNGKey(0))[0]
                           - policy(o, jax.random.PRNGKey(0))[0]).max())
        info["reload_vs_memory_action_maxdev"] = dev
        print(f"[teacher] pickle reload vs in-memory policy: max |da| = {dev:.2e}", flush=True)
        assert dev < 1e-4, "reloaded teacher differs from the trained policy"
    CM.write_json(info_path, info)
    report["teacher_info"] = {k: v for k, v in info.items()}

    # ---------------- 3. evaluation at every noise level ----------------
    tr, br = tuple(spec.target_rel_slice), tuple(spec.box_rel_slice)
    report["eval"] = {}
    for sig in SIGMAS:
        t0 = time.time()
        raw = C.rollout(env, policy, keys, spec.episode_length, action_noise=sig,
                        target_rel_slice=tr, box_rel_slice=br, spec=spec)
        ep = score(raw, radius, hold)
        ep.update({"_dist": raw["dist"], "_done": raw["done"],
                   "_nonfinite": raw["nonfinite"]})
        s = summarize(ep, radius, hold)
        s.pop("per_episode_label")
        rkeep = {k: v for k, v in raw.items()
                 if k not in ("obs", "clean", "grip_dist", "dist_obs_v1")}
        save_episodes(str(CM.RESULTS / "episodes" / "teacher" / f"teacher__sigma{sig:g}.npz"),
                      keys, rkeep, ep, {"sigma": sig, "noise_key": "PRNGKey(0)",
                                        "params_sha256": info["params_sha256"],
                                        "protocol": MT.PROTOCOL_VERSION})
        report["eval"][f"{sig:g}"] = s
        print(f"[teacher-eval] sigma={sig:g} success={s['n_success']}/{s['n_episodes']} "
              f"term={s['terminated_frac']:.2f} changed_vs_v1={s['n_changed_vs_v1_same_rollout']} "
              f"contact={s['contact_frac']:.2f} lifted={s['lifted_frac']:.2f} "
              f"within12mm={s['within_12mm_frac']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- 4. dataset audit ----------------
    report["datasets"] = {}
    vreset = jax.jit(jax.vmap(env.reset))
    for sig, fn in DATASETS.items():
        d = DemoSet.load(str(CM.DATA / fn))
        m = d.meta
        idx = np.asarray(m["subset_episode_indices"])
        assert idx.max() < 512
        ckeys = jax.random.split(jax.random.PRNGKey(int(m["collect_key_base"])), 512)[idx]
        obs0 = np.asarray(vreset(ckeys).obs)
        starts = d.episode_starts
        reset_dev = float(np.abs(obs0 - d.obs[starts]).max())
        # v2 relabel from stored states 0..n-1 (the terminal state is not stored)
        relab = []
        for i in range(d.n_episodes):
            dist = np.linalg.norm(d.obs[d.episode_slice(i)][:, tr[0]:tr[1]], axis=1)
            relab.append(bool(MT.longest_run((dist < radius)[None])[0] >= hold))
        relab = np.asarray(relab)
        # new teacher's action at every stored state vs the stored action
        ta = np.concatenate([np.asarray(policy(jp.asarray(d.obs[i:i + 4096]),
                                               jax.random.PRNGKey(0))[0])
                             for i in range(0, len(d.obs), 4096)])
        resid = d.actions - np.clip(ta, -1, 1)
        L = d.episode_lengths
        rec = {
            "file": f"data/{fn}", "sha256": CM.sha256_file(CM.DATA / fn),
            "n_episodes": d.n_episodes, "n_transitions": int(len(d.obs)),
            "collect_key_base": int(m["collect_key_base"]),
            "filtered_to_success": bool(m.get("filtered_to_success", False)),
            "raw_pool_episodes": m.get("raw_pool_episodes"),
            "raw_pool_success_rate_v1": m.get("raw_pool_success_rate"),
            "stored_success": int(d.success.sum()),
            "v2_relabel_success": int(relab.sum()),
            "v2_relabel_disagreements": int((relab != d.success).sum()),
            "episodes_terminated_early": int((L < spec.episode_length).sum()),
            "reset_obs_vs_recorded_keys_maxdev": reset_dev,
            "new_teacher_action_residual_mean_abs": float(np.abs(resid).mean()),
            "new_teacher_action_residual_std": float(resid.std()),
            "new_teacher_step0_action_maxdev": float(np.abs(resid[starts]).max()),
        }
        report["datasets"][f"{sig:g}"] = rec
        print(f"[audit] {fn}: reset dev {reset_dev:.1e}, v2 relabel "
              f"{rec['v2_relabel_success']}/{d.n_episodes} (stored {rec['stored_success']}), "
              f"residual |a_stored - a_newteacher| mean {rec['new_teacher_action_residual_mean_abs']:.3f}",
              flush=True)
    report["eval_reset_keys_sha256"] = CM.sha256_json(key_data(keys).tolist())
    report["protocol"] = MT.PROTOCOL_VERSION
    CM.write_json(report_path, report)
    print("STAGE1R_DONE", flush=True)


if __name__ == "__main__":
    main()
