"""Shared paths and run-identity helpers for every stage script.

Paths are derived from this checkout, never from a particular machine:
  PROJ      the submission root (the directory holding code/, data/, ...).
            Default: the parent of this scripts/ directory; override with $PROJ.
  HW2_RUNS  where checkpoints and other large run outputs go.
            Default: $PROJ/runs (git-ignored).

Resumable work is keyed by an IDENTITY: a hash over everything an output
depends on (input file hashes, resolved configuration, code hashes, evaluation
keys, protocol version). A stage skips work only when the stored identity of
an existing output matches the identity it would produce now -- the mere
existence of a result file is never enough.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

PROJ = Path(os.environ.get("PROJ") or Path(__file__).resolve().parents[1]).resolve()
os.environ["PROJ"] = str(PROJ)
CODE = PROJ / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
RUNS = Path(os.environ.get("HW2_RUNS") or PROJ / "runs").resolve()
DATA = PROJ / "data"
RESULTS = PROJ / "results"
CONFIGS = PROJ / "configs"
VIDEOS = PROJ / "videos"


def rel(p) -> str:
    """Path relative to PROJ (or RUNS) for records; never an absolute path."""
    p = Path(p).resolve()
    for base, tag in ((PROJ, ""), (RUNS, "$HW2_RUNS/")):
        try:
            return tag + str(p.relative_to(base))
        except ValueError:
            pass
    return p.name


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str)
                          .encode()).hexdigest()


def code_hash(files: Iterable[str]) -> str:
    """Hash of the given source files (paths relative to code/hw2)."""
    h = hashlib.sha256()
    for f in sorted(files):
        p = CODE / "hw2" / f
        h.update(f.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


TRAIN_CODE = ["train_il.py", "data.py", "policies/act.py", "policies/flow.py"]
EVAL_CODE = ["evaluate.py", "scoring.py", "metrics.py", "inference.py", "envs.py"]


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_jsonable))
    os.replace(tmp, path)


def _jsonable(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def git_describe() -> str:
    import subprocess
    try:
        sha = subprocess.run(["git", "-C", str(PROJ), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(PROJ), "status", "--porcelain",
                                "--", "code", "scripts"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "") if sha else "unknown"
    except Exception:
        return "unknown"
