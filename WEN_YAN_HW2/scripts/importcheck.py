"""compile() + real import of every module and script. ast.parse is not enough:
it accepts a misplaced `from __future__ import ...` that fails at import."""
import glob, importlib, os, sys, traceback
os.environ["JAX_PLATFORMS"] = "cpu"
PROJ = os.environ["PROJ"]
sys.path.insert(0, os.path.join(PROJ, "code"))

bad = []
files = sorted(glob.glob(os.path.join(PROJ, "code", "hw2", "**", "*.py"), recursive=True)) \
      + sorted(glob.glob(os.path.join(PROJ, "scripts", "*.py")))
for f in files:
    src = open(f).read()
    try:
        compile(src, f, "exec")
    except SyntaxError as e:
        bad.append((os.path.relpath(f, PROJ), f"COMPILE {e.msg} line {e.lineno}"))
        continue
    print(f"  compile OK  {os.path.relpath(f, PROJ)}")

mods = ["hw2", "hw2.data", "hw2.envs", "hw2.collect", "hw2.teacher",
        "hw2.train_il", "hw2.evaluate", "hw2.aggregate", "hw2.figures",
        "hw2.report", "hw2.policies.act", "hw2.policies.flow"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  import OK   {m}")
    except Exception as e:
        bad.append((m, f"IMPORT {type(e).__name__}: {e}"))
        traceback.print_exc()

if bad:
    print("\nFAILURES:")
    for n, why in bad:
        print(f"  {n}: {why}")
    sys.exit(1)
print(f"IMPORTCHECK_OK ({len(files)} files compiled, {len(mods)} modules imported)")
