# CSCI 645 HW2 — completion report (revision, protocol v2)

Branch `submission/wen-yan-hw2` of `ywen0511/CSCI645-HW2`. Starting commit:
`c59019f` (original submission `b557dab` + `HW2_AGENT_PROMPT.md`).
Submission folder: `WEN_YAN_HW2/` (its `README.md` has every command).

## Status

| Item | Status |
|---|---|
| Termination-aware metrics, 17 unit tests | done (`code/hw2/metrics.py`, `scoring.py`, `tests/`) |
| Failure taxonomy rebuilt on simulator evidence | done; thresholds checked on 814 successful episodes and on videos |
| 21 students retrained | done (originals unrecoverable) |
| Teacher retrained, evaluated at σ ∈ {0, .1, .3, .5, 1, 2}, datasets audited | done |
| Suite A: 21 models × {ensemble, replan1} × 100 original ICs (4,200 episodes) | done |
| Suite B: frozen protocol, 6 models × 100 fresh ICs (600 episodes) | done; protocol pushed in `1f31b7b` before any evaluation |
| Report, figures, tables, videos regenerated; PDF read back page by page | done (`WEN_YAN_HW2/report.pdf`, 13 pages) |
| Fresh-checkout rebuild (GitHub clone, cluster CPU node, pdflatex) | done: tests pass, regenerated text artifacts byte-identical |
| ZIP built from the commit, extracted and rebuilt | done: outputs byte-identical |
| GitHub release with ZIP + checkpoint bundles | **not done — needs your approval** (see below) |

## What was fixed

1. **Evaluation termination.** Rollouts kept stepping after `done`; v1 scored
   success over post-termination states. v2 masks the terminal state and
   everything after it, distinguishes horizon / out-of-bounds / non-finite,
   counts only the terminating transition's finite reward, and uses the
   simulator's box/target positions. Applied to teacher, students, datasets,
   diagnostics and videos. **Effect: 0 of 4,200 suite-A outcomes change** when
   the same rollouts are scored with the v1 rule.
2. **Failure analysis.** The v1 classifier tested the 12 mm proximity before
   success: 14 of 27 successful baseline episodes had failure labels. Labels are
   now success-first and mutually exclusive (terminated / never touched /
   touched, not lifted / lifted, far / short hold), from MJX contacts, box
   height and distance. All 814 successful episodes show contact and ≥20.6 cm
   lift, but only 76 % come within 12 mm, so the 12 mm test is not a reach test.
3. **Reset-state quirk** (found during this revision). MJX returns the reset
   state before computing body positions (box/gripper read as 0). Position-based
   diagnostics now start at the first stepped state; all saved rollouts were
   re-scored from their stored traces (`scripts/rescore_episodes.py`).
4. **Report.** Introduction/σ description, 21 (not 18) models, the "mediocre
   teacher" reflection, conflicting 26/27/29 % figures, `/rightarrow` and
   `/sigma` strings (caused by the table escaper), quotation marks, causal
   overstatements, and an inaccurate claim that the ensemble weighting is ACT's
   (it weights the newest prediction most; ACT's order is now a diagnostic).
5. **Reproducibility.** No hard-coded cluster paths; rendering via
   `HW2_GL` / `HW2_OSMESA_PREFIX`; pinned setup script + lock file; resumable
   steps keyed by content hashes; CPU-only report build from committed files;
   ZIP from `git archive`.

## Results (protocol v2; success %, mean of 3 seeds, IC-bootstrap 95 % CI)

Suite A — original 100 held-out ICs (per-seed values in `results/summary.csv`):

| Condition | Teacher (retrained / original) | Ensembling | Replan every step |
|---|---:|---:|---:|
| ACT, 25 clean | 100 / 100 | 10.7 [7.0, 14.7] | 9.0 [5.3, 13.0] |
| ACT, 50 clean | 100 / 100 | 14.0 [9.0, 19.7] | 18.7 [13.0, 25.0] |
| ACT, 100 clean (baseline) | 100 / 100 | 24.3 [18.0, 31.0] | 35.3 [28.3, 42.3] |
| ACT, 100 at σ=0.5 | 65 / 70 | 20.7 [15.3, 26.3] | 67.0 [59.7, 74.0] |
| ACT, 100 at σ=1 | 0 / 1 | 0.0 | 1.0 [0.0, 2.3] |
| ACT, 100 at σ=2 | 0 / 0 | 0.0 | 0.0 |
| Flow matching, 100 clean (suppl.) | 100 / 100 | 24.3 [17.7, 31.3] | 14.0 [9.3, 19.0] |

Key contrasts (pp): 25→100 clean +13.7 [+7.0, +20.7] (ens.), +26.3 [+19.0, +33.7]
(replan); clean→σ=0.5 −3.7 [−11.7, +4.0] (ens., not significant), +31.7
[+22.0, +41.0] (replan).

Suite B — 100 fresh ICs (`PRNGKey(17777)`), replanning, frozen protocol:

| | Seed 0 | Seed 1 | Seed 2 | Mean [IC CI] |
|---|---:|---:|---:|---:|
| ACT, 100 clean | 21 | 20 | 20 | 20.3 [14.7, 26.3] |
| ACT, 100 at σ=0.5 | 70 | 67 | 65 | 67.3 [59.7, 74.7] |
| Difference (pp) | +49 | +47 | +45 | **+47.0 [+39.0, +54.7]**, two-way [+37.7, +56.0] |

**Confirmed** under the frozen decision rule. The clean baseline does worse on
the fresh ICs than on suite A (20.3 vs 35.3 %, non-overlapping IC intervals),
so the original 100 ICs were easier than average for it; the σ=0.5 students
score the same on both sets.

Old vs new: retrained per-seed results are close to the archived v1 numbers
(e.g. baseline ensembling 27/22/29 → 26/19/28; σ=0.5 replan 73/64/63 →
70/67/64). Because the checkpoints differ, these differences are not
attributed to the evaluation fix; the fix itself changes no outcome.

## Limitations

- Three training seeds per condition; the two-way intervals are crude.
- The retrained teacher is **not** the data-collecting teacher (its actions
  differ from the stored ones by 0.35 on average), so teacher-referenced
  diagnostics are approximate; teacher success under v2 is reported for the
  retrained teacher, the original teacher's numbers are archived v1 values.
- Batched GPU evaluation is not bit-deterministic: re-running an evaluation
  reproduces 96–97 of 100 outcomes. Videos therefore come from one rollout and
  are labelled by that rollout, never by a different run.
- The noise ablation confounds action quality with state coverage; covariate
  shift is the best-supported explanation, not an isolated mechanism.

## Measured runtimes (A100-80GB, gc216/gc217)

PPO teacher 60M steps: 11 min. Student: 4.3 min (ACT) / 1.5 min (flow), eight
concurrent on four GPUs (~15 min for all 21). Evaluation: ~54 s per model,
mode and 100 ICs. Report rebuild: ~1 min + LaTeX.

## Artifacts

- Branch: https://github.com/ywen0511/CSCI645-HW2/tree/submission/wen-yan-hw2
- Report: `WEN_YAN_HW2/report.pdf` on the branch.
- **Submission ZIP** `WEN_YAN_HW2.zip` (52.7 MB, `git archive` of `WEN_YAN_HW2/`
  at commit `e8e3db40ed2f2e1a754de24170584c9d380ffa04`, tree `69cb8bc5…`),
  SHA-256 `e22dcb56618622aa7984af000e21ae759878b0fefcc0623e4c97c441a82659b6`.
  Rebuild it identically with `python WEN_YAN_HW2/scripts/package.py --rev e8e3db4`.
- Checkpoint bundles (SHA-256 also in `WEN_YAN_HW2/results/checkpoints_manifest.json`):
  - `hw2_ckpt_act_quantity.tar` (917 MB) `fa4330ae86add7949080a33c1987208d266a105e1756ba3a587bcecdf61db167`
  - `hw2_ckpt_act_noise.tar` (917 MB) `5dff5befa0d8bc4d430bec2e87784efcc0d22249a86925a41f4e51b8bb3739d5`
  - `hw2_ckpt_flow_and_teacher.tar` (124 MB) `da8c2b3ebb2c3c25ec846206301735e6ac0cb8fa326587352784bb127877f6ef`
- Where they are now: the GPU cluster filer (`kztrgg/runs/csci645-hw2/release/`)
  and the maintainer's laptop. They are **not yet downloadable from GitHub**:
  creating the release was blocked pending explicit approval. `WEN_YAN_HW2/README.md`
  already refers to the release name. To publish (repository visibility is
  unchanged; the release is public because the repository is public):

  ```bash
  gh release create hw2-v2-checkpoints -R ywen0511/CSCI645-HW2 \
    --target e8e3db40ed2f2e1a754de24170584c9d380ffa04 \
    --title "HW2 revision (protocol v2): submission ZIP and checkpoints" \
    --notes-file notes.md WEN_YAN_HW2.zip SHA256SUMS.txt hw2_ckpt_*.tar
  ```

## Reproduction (short form; full list in `WEN_YAN_HW2/README.md`)

```bash
cd WEN_YAN_HW2
ENVP=$PWD/.venv-hw2 OSMESA_PREFIX=$PWD/.osmesa bash scripts/setup_env.sh
export HW2_PYTHON=$PWD/.venv-hw2/bin/python
. scripts/env.sh && $PY -m pytest tests -q
scripts/run.sh scripts/stage1r_teacher.py
scripts/run.sh scripts/stage2_train.py
scripts/run.sh scripts/stage2_eval.py --suite A
scripts/run.sh scripts/stage2_eval.py --suite B
HW2_GL=osmesa HW2_OSMESA_PREFIX=$PWD/.osmesa scripts/run.sh scripts/stage2b_failure.py
scripts/run.sh scripts/diag_student.py && scripts/run.sh scripts/diag_replay.py
scripts/run.sh scripts/capture_env.py && scripts/run.sh scripts/collect_run_artifacts.py --bundle
scripts/run.sh scripts/stage3_report.py       # CPU only
scripts/run.sh scripts/package.py
```

The final course submission to Brightspace is left to the author.
