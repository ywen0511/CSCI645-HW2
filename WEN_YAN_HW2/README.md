# CSCI 645 Homework 2 — Imitation Learning for Robot Manipulation

ACT (Action Chunking with Transformers) trained on PPO-teacher demonstrations in
**`PandaPickCube`** (MuJoCo Playground), with two controlled demonstration-data
experiments: **Option A** (quantity: 25 / 50 / 100 demos) and **Option B**
(teacher action noise σ ∈ {0, 0.1, 0.3}). A flow-matching policy is included as
an additional modern baseline.

Everything in `report.pdf` is generated from the stored result files — every
number in the prose is a LaTeX macro computed by `stage3_report.py`.

---

## 1. Layout

```
code/hw2/
  envs.py          PandaPickCube wrapper, success criterion, OSMesa rendering
  data.py          DemoSet storage, normalization, action-chunk dataset
  collect.py       batched teacher rollouts (demo collection + teacher eval)
  teacher.py       brax PPO teacher
  policies/act.py  ACT (transformer CVAE, action chunking)   <- required method
  policies/flow.py flow-matching policy, DiT/adaLN-Zero      <- extra baseline
  train_il.py      imitation-learning training loop
  evaluate.py      closed-loop evaluation (chunk / temporal-ensemble inference)
  aggregate.py     seed pooling, Wilson CIs, paired bootstrap + McNemar
  figures.py       all report figures
  report.py        LaTeX macro and table generation
scripts/
  env.sh                      shared environment variables
  run.sh                      entry point (sources env.sh, runs python)
  submit.sh                   LSF submission (single node, self-terminating)
  capture_env.py              records versions/hardware -> results/environment.json
  stage1_teacher_collect.py   teacher + datasets + expert videos
  stage2_run_conditions.py    train + evaluate one shard of the condition grid
  stage3_report.py            aggregate -> figures/tables -> report.pdf
  smoke.py, bench.py          environment checks / throughput benchmark
configs/      resolved configuration snapshots
data/         demonstration datasets (.npz)
results/      conditions/*.json, summary.csv, figures/, tables/
videos/       expert and student rollouts
report/       report_template.tex, numbers.tex (generated), report.pdf
```

## 2. Environment setup

Python 3.12. **The version pins matter** — see the two traps below.

```bash
conda create -y -p $ENVP python=3.12
$ENVP/bin/pip install "jax[cuda12]==0.9.2" flax==0.11.2
$ENVP/bin/pip install "git+https://github.com/google-deepmind/mujoco_playground@d59156e099a8785dd58fbce2e25a6ee1f7b5800c"
$ENVP/bin/pip install brax==0.14.2
$ENVP/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
$ENVP/bin/pip install mediapy "imageio[ffmpeg]" matplotlib pandas scipy tqdm einops h5py tensorboard
# software GL for rendering (no NVIDIA graphics userspace on our GPU nodes):
conda create -y -p $BASE/envs/osmesa_hw2 -c conda-forge mesalib=23.0.1
```

Exact versions used for the reported results are in `results/environment.json`.

### Two traps, both of which fail late and confusingly

1. **`brax` 0.14.2 needs an older `jax`.** It still calls
   `jax.device_put_replicated`, which newer jax removed; with `jax` 0.11.x the
   failure appears only when PPO actually starts
   (`AttributeError: jax.device_put_replicated is deprecated`). `flax` must come
   down with it. Working triple: **jax/jaxlib 0.9.2, flax 0.11.2, brax 0.14.2.**
2. **No EGL on these GPU nodes.** The assignment's snippet uses
   `MUJOCO_GL=egl`, which cannot work where there is no NVIDIA graphics
   userspace. Rendering goes through software OSMesa, configured by
   `hw2.envs.use_osmesa()` **before `import mujoco`** — necessary because
   PyOpenGL resolves `libOSMesa` via `ctypes.util.find_library`, which does
   *not* consult `LD_LIBRARY_PATH`; when it fails, PyOpenGL loads GL as `None`
   and even `import mujoco` dies with
   `AttributeError: 'NoneType' object has no attribute 'glGetError'`.
   `mediapy` additionally needs an `ffmpeg` binary, which we take from the
   `imageio-ffmpeg` wheel (`hw2.envs.ensure_ffmpeg()`).

## 3. Reproducing the results

```bash
export PROJ=/path/to/csci645-hw2
. scripts/env.sh

# Stage 1: PPO teacher, teacher evaluation, demonstration pools, expert videos
scripts/run.sh scripts/stage1_teacher_collect.py \
    --out $PROJ/runs/teacher --num_timesteps 60000000 \
    --n_eval 100 --n_demo_target 100 --collect_batch 512 \
    --noise_levels 0.0,0.1,0.3

# Stage 2: every condition (18 cells = 6 conditions x 3 seeds); shard over GPUs
for S in 0 1 2 3; do
  scripts/run.sh scripts/stage2_run_conditions.py --shard $S --n_shards 4 \
      --steps 6000 --chunk 30 --batch_size 128 --lr 1e-4 \
      --n_eval 100 --exec_mode ensemble &
done; wait

# Stage 3: aggregate, figures, tables, report.pdf  (CPU only)
scripts/run.sh scripts/capture_env.py
scripts/run.sh scripts/stage3_report.py
```

On this cluster each stage is submitted with `scripts/submit.sh <name> <ngpu>
<wall_min> <slots> ...`, which pins the job to one node and makes the workload
*be* the job, so no GPU is ever held idle.

## 4. Seeds

| Purpose | Seed |
|---|---|
| Student training (init, shuffling, CVAE sampling) | `0, 1, 2` |
| Held-out evaluation initial conditions (shared by **all** conditions, and by the teacher) | `jax.random.PRNGKey(7777)`, split into 100 |
| Demonstration collection | `jax.random.PRNGKey(1000 + floor(1000σ) + batch)` |
| PPO teacher | `0` |

Evaluation seeds are disjoint from demonstration-collection seeds, so no student
is evaluated on an initial condition it trained on.

## 5. Success criterion (defined by us — the environment has none)

`PandaPickCube` exposes no success flag, and its `done` fires only on
out-of-bounds/NaN, so episodes always run the full 150 steps. We define:

> **success** = box within **5 cm** of the target for **10 consecutive control
> steps** (0.2 s).

The hold requirement prevents a fly-through from counting. `results/` reports
success as a function of the radius (3/5/8/10/15 cm) so the threshold
sensitivity is visible.

## 6. Notes on what is self-authored

`code/hw2/**` is written for this assignment. ACT follows Zhao et al. (2023)
and the flow-matching policy follows Lipman et al. (2023) / Peebles & Xie
(2022); neither upstream repository is vendored. The environment, its PPO
hyperparameters, and the random-action snippet come from MuJoCo Playground at
the commit pinned above. See the citations section of `report.pdf`, which also
states how an LLM was used.
