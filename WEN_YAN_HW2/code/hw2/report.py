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


def esc(s: str) -> str:
    """Escape PLAIN text (file names, labels without math) for LaTeX.

    Never feed LaTeX markup through this: the historical version replaced every
    backslash with '/', which is how '/rightarrow' and '/sigma' ended up in the
    rendered tables. Labels that contain markup are passed through unchanged.
    """
    rep = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
           "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}"}
    return "".join(rep.get(ch, ch) for ch in s)


def pct(x: float, nd: int = 1) -> str:
    return f"{100.0 * x:.{nd}f}"


def ci(c, nd: int = 1) -> str:
    return f"[{pct(c[0], nd)}, {pct(c[1], nd)}]"


def write_table(path: str, colspec: str, header: List[str], rows: List[List[str]],
                caption: str, label: str, size: str = r"\small",
                midrules: Optional[List[int]] = None, star: bool = False) -> None:
    """A complete float environment (never a fragment inside a tabular)."""
    env = "table*" if star else "table"
    lines = [rf"\begin{{{env}}}[t]", r"\centering", size,
             r"\begin{adjustbox}{max width=\linewidth}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if midrules and i in midrules:
            lines.append(r"\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              rf"\caption{{{caption}}}",
              rf"\label{{{label}}}", rf"\end{{{env}}}"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
