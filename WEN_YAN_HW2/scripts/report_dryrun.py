"""Build the whole report from SYNTHETIC results, to validate stage 3 (and the
LaTeX) before the real numbers exist. CPU only."""
import json, os, shutil, subprocess, sys
import numpy as np
os.environ["JAX_PLATFORMS"] = "cpu"

REAL = os.environ["PROJ"]
FAKE = os.path.join(REAL, "dryrun_report")
shutil.rmtree(FAKE, ignore_errors=True)
for d in ("results/conditions", "runs/teacher", "runs/conditions", "data",
          "videos", "report", "configs"):
    os.makedirs(os.path.join(FAKE, d), exist_ok=True)
shutil.copytree(os.path.join(REAL, "code"), os.path.join(FAKE, "code"))
shutil.copy(os.path.join(REAL, "report", "report_template.tex"),
            os.path.join(FAKE, "report", "report_template.tex"))
if os.path.exists(os.path.join(REAL, "results", "environment.json")):
    shutil.copy(os.path.join(REAL, "results", "environment.json"),
                os.path.join(FAKE, "results", "environment.json"))
sys.path.insert(0, os.path.join(FAKE, "code"))
rng = np.random.default_rng(0)
SPEC = {"obs_dim": 66, "act_dim": 8, "episode_length": 150, "ctrl_dt": 0.02,
        "nq": 16, "nv": 15, "target_rel_slice": [49, 52],
        "gripper_slice": [31, 34], "box_rel_slice": [46, 49]}
NEVAL, SEEDS = 100, (0, 1, 2)
RADIUS, HOLD = 0.05, 10

# ---- synthetic demo datasets ----
from hw2.data import DemoSet
datasets = {}
for tag, sig, rate in (("sigma0", 0.0, 1.0), ("sigma0.1", 0.1, 0.62), ("sigma0.3", 0.3, 0.24)):
    n_ep, T = 100, 150
    obs = rng.normal(size=(n_ep * T, 66)).astype(np.float32)
    # make the decoded positions plausible
    obs[:, 31:34] = rng.normal(0, 0.1, (n_ep * T, 3)) + np.array([0.4, 0.0, 0.3])
    obs[:, 46:49] = rng.normal(0, 0.05, (n_ep * T, 3))
    obs[:, 49:52] = rng.normal(0, 0.15, (n_ep * T, 3))
    ds = DemoSet(obs=obs,
                 actions=np.clip(rng.normal(size=(n_ep * T, 8)), -1, 1).astype(np.float32),
                 rewards=rng.normal(8, 1, n_ep * T).astype(np.float32),
                 episode_ends=np.arange(1, n_ep + 1) * T,
                 success=(rng.random(n_ep) < rate),
                 meta={"name": tag, "action_noise": sig,
                       "success_radius": RADIUS, "success_hold": HOLD,
                       "filtered_to_success": tag == "sigma0",
                       "raw_pool_episodes": 256, "raw_pool_success_rate": rate})
    p = os.path.join(FAKE, "data", f"demos_{tag}.npz"); ds.save(p)
    datasets[tag] = {"path": p, "summary": ds.summary(), "meta": ds.meta}

# ---- synthetic teacher report ----
teacher_eval = {}
for sig, sr in ((0.0, 0.71), (0.1, 0.52), (0.3, 0.18)):
    teacher_eval[f"teacher_eval_sigma{sig}"] = {
        "n_episodes": NEVAL, "n_success": int(sr * NEVAL), "success_rate": sr,
        "success_ci95": [max(0.0, sr - .09), min(1.0, sr + .09)],
        "min_dist_mean": 0.04 + sig, "final_dist_mean": 0.09 + sig,
        "return_mean": 1400 - 900 * sig, "return_std": 120.0,
        "out_of_bounds_frac": 0.01, "reached_box_frac": 0.95 - sig,
        "min_grip_dist_mean": 0.01, "success_radius": RADIUS, "success_hold": HOLD,
        "success_by_radius": {"0.03": sr * .7, "0.05": sr, "0.08": sr * 1.1,
                              "0.1": sr * 1.15, "0.15": sr * 1.2},
        "min_dist_percentiles": {"p10": 0.01, "p25": 0.02, "p50": 0.04,
                                 "p75": 0.09, "p90": 0.2}}
s1 = {"task": "PandaPickCube", "spec": SPEC, "seed": 0, "teacher_wall_s": 900.0,
      "ppo_params": {"num_timesteps": "60000000", "num_envs": "2048"},
      "success_radius": RADIUS, "success_hold": HOLD, "snapshots": [0, 21299200],
      "datasets": datasets, **teacher_eval,
      "expert_videos": []}
json.dump(s1, open(os.path.join(FAKE, "runs", "teacher", "stage1_report.json"), "w"), indent=2)
json.dump([{"step": s, "wall_s": w, "eval/episode_reward": r,
            "eval/episode_reward_std": 100.0}
           for s, w, r in [(0, 35, 99.5), (21299200, 300, 1358.9),
                           (42598400, 470, 1414.2), (60000000, 640, 1442.0)]],
          open(os.path.join(FAKE, "runs", "teacher", "ppo_progress.json"), "w"), indent=2)

# ---- synthetic per-condition results + train logs ----
def mk(group, algo, n, sig, seed, rate):
    suc = (rng.random(NEVAL) < rate)
    name = (f"{algo}_clean_n{n}_s{seed}" if sig == 0 else f"{algo}_noise{sig:g}_n{n}_s{seed}")
    rec = {"name": name, "algo": algo, "demos": "x.npz", "n_demos": n, "seed": seed,
           "group": group, "successful_only": sig == 0, "noise": sig,
           "exec_mode": "ensemble", "chunk": 30, "policy_overrides": {},
           "train_steps": 6000, "batch_size": 128, "lr": 1e-4,
           "n_params": 8123456, "demo_summary": datasets["sigma0"]["summary"],
           "eval": {"n_episodes": NEVAL, "n_success": int(suc.sum()),
                    "success_rate": float(suc.mean()),
                    "success_ci95": [float(max(0, suc.mean()-.09)), float(min(1, suc.mean()+.09))],
                    "min_dist_mean": 0.05, "final_dist_mean": 0.1,
                    "return_mean": 1200.0, "return_std": 150.0,
                    "out_of_bounds_frac": 0.02, "reached_box_frac": 0.9,
                    "min_grip_dist_mean": 0.011,
                    "success_radius": RADIUS, "success_hold": HOLD,
                    "per_episode_success": suc.astype(bool).tolist()},
           "wall_s": 300.0, "eval_seed_base": 7777, "n_eval": NEVAL}
    json.dump(rec, open(os.path.join(FAKE, "results", "conditions", name + ".json"), "w"), indent=2)
    rd = os.path.join(FAKE, "runs", "conditions", name); os.makedirs(rd, exist_ok=True)
    json.dump({"history": [{"step": s, "loss": 30*np.exp(-s/2000)+1, "l1": 1.5*np.exp(-s/2500)+0.1,
                            "kl": 2.0, "lr": 1e-4, "grad_norm": 1.0, "wall_s": s/20}
                           for s in range(0, 6000, 200)],
               "config": {"algo": algo}, "wall_s": 300.0},
              open(os.path.join(rd, "train_log.json"), "w"), indent=2)

for n, rate in ((25, 0.30), (50, 0.45), (100, 0.58)):
    for s in SEEDS: mk("quantity", "act", n, 0.0, s, rate)
for sig, rate in ((0.1, 0.44), (0.3, 0.17)):
    for s in SEEDS: mk("noise", "act", 100, sig, s, rate)
for s in SEEDS: mk("algorithm", "flow", 100, 0.0, s, 0.55)

# ---- synthetic frames + failure analysis ----
for i in range(2):
    fr = (rng.random((8, 60, 80, 3)) * 255).astype(np.uint8)
    np.savez_compressed(os.path.join(FAKE, "videos", f"expert_demo{i}_frames.npz"),
                        frames=fr, frame_index=np.arange(8) * 9, n_total=75)
d_s = np.maximum(0.01, 0.4 - 0.4 * np.linspace(0, 1, 151)[None, :] + rng.normal(0, .01, (40, 151)))
d_f = np.maximum(0.08, 0.4 - 0.25 * np.linspace(0, 1, 151)[None, :] + rng.normal(0, .02, (60, 151)))
json.dump({"condition": "act_clean_n100_s0", "n_eval": NEVAL,
           "success_radius": RADIUS, "success_hold": HOLD, "success_rate": 0.4,
           "reached_box_frac": 0.9,
           "dist_success": d_s.tolist(), "dist_failure": d_f.tolist(),
           "min_dist_all": np.concatenate([d_s.min(1), d_f.min(1)]).tolist(),
           "min_grip_all": rng.random(100).tolist(),
           "failure_taxonomy": {"reached box, never lifted": 31,
                                "lifted, missed target": 18,
                                "entered target, did not hold": 8,
                                "never reached the box": 3},
           "labels": [], "success_by_radius": {"0.03": .3, "0.05": .4, "0.08": .5,
                                               "0.1": .55, "0.15": .6}},
          open(os.path.join(FAKE, "results", "failure_analysis.json"), "w"), indent=2)

print("synthetic inputs written to", FAKE)
env = dict(os.environ); env["PROJ"] = FAKE
r = subprocess.run([sys.executable, os.path.join(REAL, "scripts", "stage3_report.py"),
                    "--teacher_dir", os.path.join(FAKE, "runs", "teacher")],
                   env=env, capture_output=True, text=True)
print(r.stdout[-6000:])
if r.returncode != 0:
    print("STDERR:", r.stderr[-3000:])
    print(f"REPORT_DRYRUN_FAILED rc={r.returncode}")
    sys.exit(r.returncode)
print("REPORT_DRYRUN_OK")
