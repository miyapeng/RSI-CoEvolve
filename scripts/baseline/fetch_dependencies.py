"""Fetch only missing pinned source dependencies; never update an existing checkout."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = {
    "slime": ("THUDM/slime", "0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e"),
    "SWE-bench_Pro-os": ("scaleapi/SWE-bench_Pro-os", "66f92766bba642462d4bbe5479e83f91f9211862"),
    "Vision2Web": ("zai-org/Vision2Web", "d80a4a3d0ea7326541ad87314294269a633789e4"),
    "OSWorld": ("xlang-ai/OSWorld", "8903ebeb0efdf74331bd7dc37ed91ccbbff22317"),
    "arc-agi-3-benchmarking": ("arcprize/arc-agi-3-benchmarking", "353d38be471395b5ae47dbb44df95440677397e1"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-final-eval", action="store_true")
    parser.add_argument("--ssh", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--benchmarks", nargs="+", choices=["Vision2Web", "OSWorld", "arc-agi-3-benchmarking"], default=[])
    a = parser.parse_args()
    names = (["slime", "SWE-bench_Pro-os"] if a.with_final_eval else ["slime"]) + a.benchmarks
    for name in names:
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
