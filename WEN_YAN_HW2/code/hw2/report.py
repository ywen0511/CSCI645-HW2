"""Generate report.tex from the stored results.

Discipline (learned the hard way):
  * Every quantity in the prose is a LaTeX macro whose value is computed here
    from the result JSON. No number is typed into the prose.
  * Macro names contain LETTERS ONLY -- `\\newcommand{\\succ100}` is not a legal
    TeX control sequence and fails in confusing ways, so digits are spelled out.
  * Tables are emitted as complete float environments, never as fragments to be
    \\input inside someone else's tabular (which silently opens a TeX group).
  * A provenance table lists the source file of every figure and table.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

_DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
           "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
           ".": "Pt", "-": "Neg", "_": ""}


def macro_name(key: str) -> str:
    """Letters-only TeX control sequence name."""
    out = []
    for ch in key:
        if ch.isalpha():
            out.append(ch)
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
    return "".join(out)


class Macros:
    def __init__(self):
        self._m: Dict[str, str] = {}

    def set(self, key: str, value) -> str:
        name = macro_name(key)
        if isinstance(value, float):
            s = f"{value:.4g}"
        else:
            s = str(value)
        self._m[name] = s
        return "\\" + name

    def pct(self, key: str, frac: float, nd: int = 1) -> str:
        return self.set(key, f"{100.0*frac:.{nd}f}")

    def tex(self) -> str:
        lines = ["% auto-generated: do not edit"]
        for k, v in sorted(self._m.items()):
            lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
        return "\n".join(lines) + "\n"


def provenance_table(entries: List[Dict[str, str]], path: str) -> None:
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\begin{tabular}{p{0.30\linewidth}p{0.62\linewidth}}", r"\toprule",
             r"Artifact & Produced from \\", r"\midrule"]
    for e in entries:
        lines.append(f"\\texttt{{{_esc(e['artifact'])}}} & \\texttt{{{_esc(e['source'])}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Provenance of every figure and table in this report.}",
              r"\label{tab:prov}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _esc(s: str) -> str:
    return (s.replace("\\", "/").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def dataset_table(datasets: Dict[str, dict], path: str) -> None:
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Dataset & $\sigma$ & Episodes & Transitions & Len. (mean$\pm$sd) & "
             r"Success \\", r"\midrule"]
    for tag in sorted(datasets):
        d = datasets[tag]; s = d["summary"]; m = d["meta"]
        lines.append(
            f"\\texttt{{{_esc(tag)}}} & {m.get('action_noise', 0):g} & "
            f"{s['n_episodes']} & {s['n_transitions']} & "
            f"{s['ep_len_mean']:.1f}$\\pm${s['ep_len_std']:.1f} & "
            f"{100*s['success_fraction']:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Collected demonstration datasets. `Success' is the "
              r"fraction of stored episodes meeting the success criterion; the "
              r"clean set is filtered to successful episodes, the noisy sets are "
              r"not (degraded data quality is the independent variable).}",
              r"\label{tab:data}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def comparison_table(cmps: List[dict], labels: Dict[str, str], path: str,
                     cmps_c1: Optional[List[dict]] = None) -> None:
    """Paired comparisons, reported under BOTH inference modes when available."""
    dual = bool(cmps_c1)
    byc1 = {c.get("key"): c for c in (cmps_c1 or [])}
    cols = r"llrrrr" if dual else r"llrrr"
    head = (r"Comparison & & $\Delta$ (ensembling) & $p$ & "
            r"$\Delta$ (re-plan) & $p$ \\" if dual else
            r"Comparison & & $\Delta$ success (95\% CI) & McNemar $p$ & "
            r"Significant \\")
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{" + cols + "}", r"\toprule", head, r"\midrule"]
    for c in cmps:
        if "error" in c:
            continue
        key = c.get("key", "")
        row = f"{_esc(labels.get(key, key))} & & "
        row += (f"{100*c['delta']:+.1f}\\% "
                f"[{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}] & "
                f"{c['mcnemar_p']:.3g} & ")
        if dual:
            o = byc1.get(key)
            if o and "error" not in o:
                row += (f"{100*o['delta']:+.1f}\\% "
                        f"[{100*o['ci95'][0]:+.1f}, {100*o['ci95'][1]:+.1f}] & "
                        f"{o['mcnemar_p']:.3g} \\\\")
            else:
                row += r"-- & -- \\\\"
        else:
            row += f"{'yes' if c['significant_at_05'] else 'no'} \\\\"
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Paired comparisons on the shared held-out initial "
              r"conditions, under both inference modes. $\Delta$ is the change "
              r"in success rate relative to the first-named condition, with a "
              r"paired bootstrap interval over episodes (20{,}000 resamples); "
              r"$p$ is McNemar\'s exact test on the seed-majority per-episode "
              r"outcomes. Note that the demonstration-quality comparisons change "
              r"sign between the two modes --- see Section~\ref{sec:fail}.}",
              r"\label{tab:paired}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def mode_table(rows_ens: List[Dict], rows_c1: List[Dict], path: str) -> None:
    """Success under ACT's temporal ensembling vs re-planning every step."""
    by = {(r["group"], r["algo"], r["n_demos"], r["noise"]): r for r in rows_c1}
    order = {"quantity": 0, "noise": 1, "algorithm": 2}
    rows = sorted(rows_ens, key=lambda r: (order.get(r["group"], 9), r["noise"],
                                           r["n_demos"]))
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Condition & Algo. & \#Demos & $\sigma$ & Ensembling & "
             r"Re-plan each step & Reached box \\", r"\midrule"]
    last = None
    for r in rows:
        if last is not None and r["group"] != last:
            lines.append(r"\midrule")
        last = r["group"]
        o = by.get((r["group"], r["algo"], r["n_demos"], r["noise"]))
        c1 = f"{100*o['success_pooled']:.1f}\\%" if o else "--"
        box = f"{100*o['reached_box_frac']:.0f}\\%" if o else "--"
        label = {"quantity": "Quantity", "noise": "Noisy teacher",
                 "algorithm": "Flow matching"}[r["group"]]
        lines.append(
            f"{label} & {r['algo'].upper()} & {r['n_demos']} & {r['noise']:g} & "
            f"{100*r['success_pooled']:.1f}\\% & {c1} & {box} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Inference mode. `Ensembling\' is ACT\'s temporal "
              r"ensembling (the pre-specified default used for the headline "
              r"numbers); `Re-plan each step\' predicts a fresh chunk every "
              r"control step and executes only its first action. `Reached box\' "
              r"is the fraction of episodes in which the gripper came within "
              r"12\,mm of the box under the re-planning mode. Every cell uses "
              r"the same held-out initial conditions.}",
              r"\label{tab:modes}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
