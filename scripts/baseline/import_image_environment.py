"""Seed a fresh Conda prefix from the already tested GPU image, without downloads.

Run once inside that image. CUDA packages are copied, not linked into another
user's environment. Slime and Megatron are installed from the separate pins.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

COPY_EXCLUDES = [
    "/pip*",
    "/setuptools*",
    "/wheel*",
    "/packaging*",
    "/_distutils_hack*",
    "/*.pth",
    "/*.egg-link",
    "/__editable__*",
    "/slime*",
    "/megatron*",
    "__pycache__",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", type=Path, required=True)
    args = p.parse_args()
    prefix = args.prefix.resolve()
    assert (prefix / "conda-meta/history").is_file(), "Create the Conda prefix first"
    site = prefix / "lib/python3.12/site-packages"
    assert not (site / "torch").exists(), "Refusing to overwrite an existing CUDA environment"
    source = Path(importlib.util.find_spec("torch").origin).parent.parent
    assert source != site
    report = {
        "image_python": sys.executable,
        "source_site": str(source),
        "prefix": str(prefix),
        "source_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "copied_source_packages": {},
    }
    command = ["rsync", "-a", "--ignore-existing"]
    command += [f"--exclude={item}" for item in COPY_EXCLUDES]
    subprocess.run([*command, str(source) + "/", str(site) + "/"], check=True)
    # The image's editable SGLang package lives outside site-packages. Keep a
    # real local package in the prefix, with source origin recorded separately.
    spec = importlib.util.find_spec("sglang")
    origin = Path(spec.origin).parent
    if origin.parent != source:
        shutil.copytree(origin, site / "sglang", dirs_exist_ok=False, ignore=shutil.ignore_patterns("__pycache__"))
        report["copied_source_packages"]["sglang"] = str(origin)
        repo = origin.parent.parent
        result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
        report["sglang_source_commit"] = result.stdout.strip() if result.returncode == 0 else None
        for name in ("LICENSE", "LICENSE.md"):
            if (repo / name).exists():
                shutil.copy2(repo / name, site / "sglang" / name)
    for ep in importlib.metadata.entry_points(group="console_scripts"):
        if ep.name in ("ray", "torchrun"):
            target = prefix / "bin" / ep.name
            target.write_text(
                f"#!{prefix}/bin/python\nimport sys\nfrom importlib.metadata import entry_points\n"
                f"sys.exit(next(e for e in entry_points(group='console_scripts') if e.name == {ep.name!r}).load()())\n"
            )
            target.chmod(0o755)
    (prefix / "image-import.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "IMAGE_PACKAGES_COPIED", "prefix": str(prefix)}), flush=True)


if __name__ == "__main__":
    main()
