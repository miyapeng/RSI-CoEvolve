"""Fetch only missing pinned source dependencies; never update an existing checkout."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = {
    "slime": ("THUDM/slime", "0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e"),
    "SWE-bench_Pro-os": ("scaleapi/SWE-bench_Pro-os", "66f92766bba642462d4bbe5479e83f91f9211862"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-final-eval", action="store_true")
    parser.add_argument("--ssh", action="store_true")
    parser.add_argument("--check", action="store_true")
    a = parser.parse_args()
    for name in ["slime", "SWE-bench_Pro-os"] if a.with_final_eval else ["slime"]:
        repo, commit = DEPENDENCIES[name]
        path = ROOT.parent / "references" / name
        if not path.exists():
            if a.check:
                parser.error(f"Missing dependency: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"git@github.com:{repo}.git" if a.ssh else f"https://github.com/{repo}.git"
            subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", url, str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", commit], check=True)
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
        if head != commit or dirty:
            parser.error(f"Existing {path} differs from the pin or is dirty; left unchanged")
        print(f"{name}: {head} (present, unchanged)")


if __name__ == "__main__":
    main()
