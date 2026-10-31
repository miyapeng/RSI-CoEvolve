"""Prepare training-side data from the fixed local baseline manifest. Never fetch final benchmarks."""
import os
from pathlib import Path
from baseline.data import tasks
from baseline.records import ROOT, append

out = Path(os.environ["SELFHPO_WORK"]) / "rl/data"
manifest = os.environ.get("BASELINE_SPLIT_MANIFEST", str(ROOT / "configs/split_manifest.json"))
for role, name in [("optimization_pool", "train.jsonl"), ("dev_eval", "dev_eval.jsonl")]:
    target = out / name
    if target.exists():
        raise FileExistsError(f"Refusing to replace {target}")
    for row in tasks(manifest, role):
        append(target, {k: row[k] for k in ("problem_statement", "patch", "metadata")})
    print(f"wrote {target}")
