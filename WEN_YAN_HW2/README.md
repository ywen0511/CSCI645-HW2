# CSCI 645 HW2 — Imitation learning on `PandaPickCube` (ACT)

ACT students trained on PPO-teacher demonstrations in MuJoCo Playground's
`PandaPickCube`, with two controlled demonstration-data experiments with the
same algorithm: **Option A** (quantity: 25 / 50 / 100 nested clean
demonstrations) and **Option B** (teacher action noise σ ∈ {0.5, 1, 2},
100 demonstrations, failures kept). A flow-matching policy on the clean 100
demonstrations is a supplementary arm. 7 conditions × training seeds {0, 1, 2}
= **21 student models**.

`report.pdf` answers the assignment questions. Every number in it is a LaTeX
macro computed by `scripts/stage3_report.py` from files in `results/`.
`COMPLETION_REPORT.md` (repository root) summarises this revision.

> **Revision note (protocol v2).** The original submission (commit `b557dab`)
> scored success over states *after* the environment had terminated and
> mislabelled successful episodes in its failure analysis. Its checkpoints were
> not recoverable. This revision retrains every model, re-evaluates everything
> with a termination-aware protocol, and adds a pre-registered confirmation on
> fresh initial conditions. The original results are archived, unchanged, in
> `results/historical_v1/`.

---

## 1. Layout

```
code/hw2/
  envs.py          env construction, rendering backend, ground-truth diagnostics
  metrics.py       termination-aware success / validity / failure taxonomy (protocol v2)
  inference.py     temporal ensembling and replanning action selection
  evaluate.py      batched closed-loop evaluation, per-episode records
  collect.py       batched teacher rollouts, dataset construction
  teacher.py       brax PPO teacher (training + reload)
  data.py          DemoSet storage, normalisation, action-chunk dataset
  grid.py          the 21-model grid and its resolved hyperparameters
  train_il.py      imitation-learning training loop
  policies/act.py  ACT (required method)       policies/flow.py  flow matching (suppl.)
  stats.py         statistics for shared initial conditions
  figures.py, report.py   figures and LaTeX tables/macros
scripts/
  setup_env.sh            create the Python environment (pinned versions)
  env.sh, run.sh          shared environment; run any stage with scripts/run.sh
  stage1r_teacher.py      retrain + evaluate the PPO teacher, audit the datasets
  stage1_collect.py       (optional) regenerate datasets with any teacher
  stage2_train.py         train the grid (resumable, identity-checked)
  freeze_fresh_protocol.py  freeze suite B (already done: configs/fresh_protocol.json)
  stage2_eval.py          suite A (all 21 x 2 modes) and suite B (fresh ICs)
  stage2b_failure.py      failure taxonomy + verified rollout videos
  diag_student.py, diag_replay.py   diagnostics used in the failure analysis
  collect_run_artifacts.py  copy run records into configs/, checkpoint manifest/bundles
  stage3_report.py        statistics, figures, tables, report.pdf (CPU only)
  package.py              build + verify WEN_YAN_HW2.zip from a commit
  jobs/*.sh, submit.sh    the LSF jobs used for the reported run (optional)
tests/            CPU unit tests (metrics, inference, statistics)
configs/          fresh_protocol.json, requirements.lock.txt, trainlogs/, identities/, teacher/
data/             the six committed demonstration datasets (.npz)
results/          eval/, episodes/, teacher/, failure/, diag/, summary.*, figures/, tables/,
                  checkpoints_manifest.json, environment.json, historical_v1/
videos/           expert demonstrations and verified student rollouts
report/           report_template.tex, numbers.tex (generated), figures/, tables/
```

## 2. Environment setup

Tested: Ubuntu 24.04, Python 3.12, NVIDIA A100 (driver in
`results/environment.json`), CUDA 12 wheels. Exact pins are in
`configs/requirements.lock.txt`.

```bash
ENVP=$PWD/.venv-hw2 OSMESA_PREFIX=$PWD/.osmesa bash scripts/setup_env.sh
export HW2_PYTHON=$PWD/.venv-hw2/bin/python
```

Two pins matter: **brax 0.14.2 needs jax/jaxlib 0.9.2 and flax 0.11.2** (it still
calls `jax.device_put_replicated`, removed in later jax; the failure appears only
once PPO starts), and **MuJoCo Playground is pinned to commit
`d59156e099a8785dd58fbce2e25a6ee1f7b5800c`** (it downloads mujoco_menagerie on
first use; the commit used is in `results/environment.json`).

**Rendering** (videos only; training/evaluation never render):

| Setting | When |
|---|---|
| `HW2_GL=egl` | machine with NVIDIA graphics userspace |
| `HW2_GL=glfw` | desktop with a display |
| `HW2_GL=osmesa HW2_OSMESA_PREFIX=<prefix>` | headless nodes without EGL (what we used); the prefix comes from `setup_env.sh` |

Optional variables: `HW2_RUNS` (checkpoints/logs, default `./runs`),
`HW2_LATEX=pdflatex|tectonic`, `HW2_TEX_BIN` (directory containing `pdflatex`).

## 3. Reproducing the results

All commands run from this directory. GPU stages need one NVIDIA GPU with
≥16 GB; the measured runtimes below are for one A100-80GB.

```bash
. scripts/env.sh
# 0. unit tests (CPU, ~1 s)
$PY -m pytest tests -q
# 1. teacher: retrain PPO (60M steps), evaluate at every sigma, audit datasets
scripts/run.sh scripts/stage1r_teacher.py --num_timesteps 60000000
# 2. students: train all 21 (resumable; shard over GPUs with --shard/--n_shards)
scripts/run.sh scripts/stage2_train.py
# 3. suite A: every model x {ensemble, replan1} on the original 100 held-out ICs
scripts/run.sh scripts/stage2_eval.py --suite A
# 4. suite B: the frozen fresh-IC confirmation (configs/fresh_protocol.json)
scripts/run.sh scripts/stage2_eval.py --suite B
# 5. failure analysis + verified videos, diagnostics, environment record
HW2_GL=osmesa HW2_OSMESA_PREFIX=... scripts/run.sh scripts/stage2b_failure.py
scripts/run.sh scripts/diag_student.py
scripts/run.sh scripts/diag_replay.py
scripts/run.sh scripts/capture_env.py
# 6. copy run records into configs/ + checkpoint manifest (and bundles)
scripts/run.sh scripts/collect_run_artifacts.py --bundle
# 7. report (CPU only; reads committed files only)
scripts/run.sh scripts/stage3_report.py
# 8. ZIP from the committed tree
scripts/run.sh scripts/package.py
```

The LSF job wrappers used for the reported run are `scripts/jobs/students.sh`
(steps 2–3, 4 GPUs), `scripts/jobs/teacher.sh` (step 1) and
`scripts/jobs/finalize.sh` (steps 4–6).

**Resuming.** Training and evaluation skip a model only when the stored
identity matches: dataset SHA-256, resolved configuration and training-code hash
for a checkpoint; checkpoint SHA-256, reset-key hash, protocol version, mode and
evaluation-code hash for an evaluation. A result file whose checkpoint is
missing or changed is recomputed, never silently reused.

**Report only, without a GPU** (from a fresh checkout or the ZIP):

```bash
python -m pip install numpy matplotlib imageio imageio-ffmpeg pypdf   # any recent versions
PROJ=$PWD python scripts/stage3_report.py      # needs pdflatex (or HW2_LATEX=tectonic)
```

This regenerates `results/summary.*`, all figures and tables, `report/numbers.tex`
and `report.pdf` from the committed results. It does not validate GPU training.

**Checkpoints.** Not stored in git. Download the three bundles from the GitHub
release `hw2-v2-checkpoints` of this repository, verify them against
`results/checkpoints_manifest.json`, and extract into `$HW2_RUNS`; steps 3–5 then
run without retraining.

## 4. Protocol

**Success** (defined by us; the environment has none): the box is within
**5 cm** of the target for **10 consecutive valid control steps** (0.2 s),
using the simulator's box and target positions.

**Termination-aware validity (protocol v2).** The environment's `done` fires
only when the box leaves the workspace or the physics becomes NaN, and it has no
auto-reset. The first terminating transition's resulting state and everything
after it are invalid and excluded from success, distances, returns (except the
terminating transition's own finite reward) and diagnostics; non-finite states
are never valid. Reaching the hold once is a success even if the box later
leaves the radius (reported separately). See `code/hw2/metrics.py` and
`tests/test_metrics.py`.

**Inference modes.** `ensemble`: temporal ensembling over all chunks covering
the current step with weights exp(−0.01·age), newest prediction weighted most
(the historical headline protocol; ACT's reference weights the oldest most, which
is evaluated as a diagnostic). `replan1`: predict a chunk every step, execute
its first action.

| Purpose | Seed / keys |
|---|---|
| Student training (init, shuffling, CVAE sampling) | `0, 1, 2` |
| Suite A initial conditions (all models, teacher) | `jax.random.split(PRNGKey(7777), 100)` |
| Suite B fresh initial conditions | `jax.random.split(PRNGKey(17777), 100)` (hash in `configs/fresh_protocol.json`) |
| Demonstration collection | `jax.random.split(PRNGKey(1000 + floor(1000σ)), 512)[i]` |
| Teacher action-noise stream | `PRNGKey(0)` |
| PPO teacher | `0` |
| Flow-matching inference noise | `torch.manual_seed(0)` before each evaluation |

Suite B keys collide with no demonstration or suite-A key (checked as raw keys
and as initial box/target configurations; see the protocol file).

## 5. What is self-authored

`code/hw2/**`, `scripts/**` and `tests/**` were written for this assignment.
ACT follows Zhao et al. (2023) and the flow-matching policy Lipman et al. (2023) /
Peebles & Xie (2023); neither upstream repository is vendored. The environment,
its PPO hyperparameters and the random-action snippet come from MuJoCo
Playground at the pinned commit. See the citations section of `report.pdf`,
which also describes how an LLM was used.
