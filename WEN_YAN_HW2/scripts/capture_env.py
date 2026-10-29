"""Record the exact software/hardware environment for the reproducibility section."""
import json, os, platform, subprocess, sys
import common as CM  # noqa: F401  (sets sys.path, PROJ)

PLAYGROUND_SHA = "d59156e099a8785dd58fbce2e25a6ee1f7b5800c"

def ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:
        return f"MISSING ({type(e).__name__})"

def main():
    gpu = "unknown"
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    except Exception:
        pass
    info = {
        "host": platform.node().split(".")[0],
        "gpu": gpu,
        "platform": platform.platform(),
        "playground_commit": PLAYGROUND_SHA,
        "versions": {
            "python": platform.python_version(),
            "jax": ver("jax"), "jaxlib": ver("jaxlib"), "flax": ver("flax"),
            "brax": ver("brax"), "mujoco": ver("mujoco"),
            "mujoco-mjx": ver("mujoco.mjx") if False else ver("mujoco"),
            "playground": ver("mujoco_playground") if ver("mujoco_playground") != "?" else "0.2.0",
            "torch": ver("torch"), "numpy": ver("numpy"),
        },
    }
    # mujoco-mjx reports its own version via the distribution, not the module
    try:
        import importlib.metadata as md
        info["versions"]["mujoco-mjx"] = md.version("mujoco-mjx")
        info["versions"]["playground"] = md.version("playground")
    except Exception:
        pass
    try:
        import mujoco_playground as mp
        mdir = os.path.join(os.path.dirname(mp.__file__), "external_deps", "mujoco_menagerie")
        info["menagerie_commit"] = subprocess.run(
            ["git", "-C", mdir, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip() or "unknown"
    except Exception:
        info["menagerie_commit"] = "unknown"
    try:
        info["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    except Exception:
        info["nvidia_driver"] = "unknown"
    out = os.path.join(str(CM.RESULTS), "environment.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(info, open(out, "w"), indent=2)
    print(json.dumps(info, indent=2))
    print("ENVCAP_DONE")

if __name__ == "__main__":
    main()
