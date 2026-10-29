# Historical results (protocol v1) — archived, superseded

These are the results of the original submission (commit `b557dab`), kept
unchanged for comparison. They were produced with **protocol v1**, which
scored success over the full 150-step trace *including states after the
environment terminated*, read the box–target distance from the observation,
and labelled failures with a classifier that tested the 12 mm gripper
proximity *before* success (so successful episodes received failure labels).
The checkpoints behind them were not recoverable; every number in the current
report comes from retrained models evaluated under protocol v2
(`results/eval/`, `results/episodes/`). Do not mix the two in one table.

| File | What it is |
|---|---|
| `conditions/*.json` | per-model v1 results (`eval` = temporal ensembling, `eval_chunk1` = per-step replanning) |
| `summary*.csv/json`, `paired_comparisons.json` | v1 aggregates |
| `failure_analysis.json` | v1 failure labels for `act_clean_n100_s0` (contains the mislabelling) |
| `diag_*.json` | v1 diagnostics |
| `teacher_stage1_report.json`, `ppo_progress.json` | the ORIGINAL teacher's training curve and evaluation (sigma 0, 0.1, 0.3) |
| `trainlogs/` | resolved configs + loss histories of the original student runs |
| `environment.json` | software versions of the original run |
