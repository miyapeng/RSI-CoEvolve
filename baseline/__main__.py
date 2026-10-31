import argparse
import json
import os
from pathlib import Path

from .audit import preflight
from .config import load_config
from .data import split, tasks
from .execution import evaluate, serve, serve_command
from .outer_loop import cross_eval, run
from .records import ROOT, write
from .report import summarize
from .reward import grade
from .search import h_step
from .training import prepare, train


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description="Self-HPO fixed-schedule 9B baseline; execution is explicit")
    p.add_argument(
        "command",
        choices=[
            "split",
            "preflight",
            "rollout",
            "oracle",
            "h-step",
            "p-step",
            "run",
            "resume",
            "cross-eval",
            "audit-holdout",
            "report",
        ],
    )
    p.add_argument("--config", default=str(ROOT / "configs/baseline_9b.yaml"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--mode", choices=["base", "h-only", "p-only", "hpo"])
    p.add_argument("--raw", type=Path)
    p.add_argument("--old-policy")
    p.add_argument("--new-policy")
    p.add_argument("--old-harness")
    p.add_argument("--new-harness")
    a = p.parse_args()
    c = load_config(a.config)
    if a.mode:
        c["h_enabled"] = a.mode in ("h-only", "hpo")
        c["p_enabled"] = a.mode in ("p-only", "hpo")
    if a.limit:
        for key in ("h_task_sample_size", "dev_task_sample_size", "p_task_sample_size"):
            c[key] = a.limit
    if a.command == "split":
        raw = a.raw or next((ROOT / "data/training/self-hpo").glob("*/raw/train.jsonl"))
        result = split(raw, a.output, c["seed"])
        print(json.dumps({"counts": result["counts"], "overlap": result["cross_split_overlap"]}, indent=2))
        return
    if a.command == "report":
        print(summarize(a.output))
        return
    report = preflight(c, phase="run" if a.command in ("resume", "preflight") else a.command)
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output / "preflight.json", report)
    if a.command == "preflight":
        print(json.dumps({"status": report["status"], "blockers": report["blockers"]}, indent=2))
        return
    rows = tasks(c["dataset_split_manifest"], "optimization_pool", a.limit or c["p_task_sample_size"])
    if a.command == "p-step" and not a.execute:
        prepare(c["model_checkpoint"], c["harness_path"], rows, c, a.output)
    else:
        write(
            a.output / "execution-plan.json",
            {
                "status": "NOT_RUN",
                "command": a.command,
                "config": c,
                "serve_command": serve_command(c["model_checkpoint"], c),
            },
        )
    if not a.execute:
        print(f"NOT_RUN: inspect {a.output}; --execute enables real resources")
        return
    if report["blockers"]:
        p.error("Preflight blocked: " + "; ".join(report["blockers"]))
    policy, harness = c["model_checkpoint"], c["harness_path"]
    if a.command in ("run", "resume"):
        run(c, a.output, resume=a.command == "resume")
        print(summarize(a.output))
    elif a.command == "p-step":
        train(policy, harness, rows, c, a.output)
    elif a.command == "oracle":
        row = rows[0]
        for name, patch in [("empty", ""), ("invalid", "not a patch"), ("reference", row["patch"])]:
            grade(row, patch, c["runtime"], a.output / name)
    elif a.command == "cross-eval":
        if not all((a.old_policy, a.new_policy, a.old_harness, a.new_harness)):
            p.error("cross-eval requires both policy and harness paths")
        cross_eval(c, a.output, a.old_policy, a.new_policy, a.old_harness, a.new_harness)
    else:
        with serve(policy, c, a.output / "serve") as client:
            if a.command == "h-step":
                opt = tasks(c["dataset_split_manifest"], "optimization_pool", c["h_task_sample_size"])
                dev = tasks(c["dataset_split_manifest"], "dev_eval", c["dev_task_sample_size"])
                h_step(client, policy, harness, opt, dev, c, a.output)
            else:
                if a.command == "audit-holdout":
                    rows = tasks(c["dataset_split_manifest"], "audit_holdout", a.limit or c["dev_task_sample_size"])
                evaluate(client, rows, policy, harness, c, a.output / "rollouts")


if __name__ == "__main__":
    main()
