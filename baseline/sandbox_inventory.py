"""Prepare image mappings without creating sandboxes or moving container images."""

import argparse
import json
from pathlib import Path

import yaml

from .config import load_config
from .data import ROLES, tasks
from .final_eval import execution_image
from .records import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/baseline_9b.yaml"))
    parser.add_argument("--split", choices=[*ROLES, "all-training"], default="optimization_pool")
    parser.add_argument("--benchmark", choices=["verified", "multilingual", "pro-v2"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        parser.error("Refusing to overwrite an image manifest")
    c = load_config(a.config)
    if a.benchmark:
        spec = yaml.safe_load((ROOT / "configs/final_eval.yaml").read_text())[a.benchmark]
        path = ROOT / "data/benchmarks" / a.benchmark / spec["revision"] / "tasks.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        images = {execution_image(a.benchmark, row) for row in (rows[: a.limit] if a.limit else rows)}
    else:
        roles = ROLES if a.split == "all-training" else [a.split]
        images = {
            row["metadata"]["image_url"] for role in roles for row in tasks(c["dataset_split_manifest"], role, a.limit)
        }
    existing = c["runtime"]["sandbox"]["images"]
    result = {"images": {name: existing.get(name, {"image": None, "digest": None}) for name in sorted(images)}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(yaml.safe_dump(result, sort_keys=True))
    print(f"{len(images)} source images listed; no images transferred or sandboxes created")


if __name__ == "__main__":
    main()
