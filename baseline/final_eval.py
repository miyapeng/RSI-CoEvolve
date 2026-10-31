"""Independent final evaluation: same agent, existing Self-HPO/official verifiers, local pinned data."""

import argparse
import importlib
import json
import os
import subprocess
from pathlib import Path

import yaml

from .audit import preflight
from .config import load_config
from .execution import serve
from .harness import snapshot
from .records import ROOT, append, digest, read, sha, write
from .runtime import run_agent
from .sandbox import TaskSandbox


def prepare(benchmark, config, output, limit=None):
    spec = yaml.safe_load((ROOT / "configs/final_eval.yaml").read_text())[benchmark]
    data = ROOT / "data/benchmarks" / benchmark / spec["revision"]
    manifest = read(data / "manifest.json")
    if (
        manifest["revision"] != spec["revision"]
        or manifest["rows"] != spec["tasks"]
        or sha(data / "tasks.jsonl") != manifest["index_sha256"]
        or sha(data / "grader.jsonl") != manifest["grader_sha256"]
    ):
        raise ValueError("Final dataset differs from its pinned manifest")
    if "verifier_repo" in spec:
        repo = (ROOT / spec["verifier_repo"]).resolve()
        if (
            subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            != spec["verifier_commit"]
        ):
            raise ValueError("Pro V2 verifier pin mismatch")
        spec["verifier_repo"] = str(repo)
    output = Path(output)
    plan = {
        "status": "NOT_RUN",
        "benchmark": benchmark,
        "dataset_revision": spec["revision"],
        "task_count": min(limit, spec["tasks"]) if limit else spec["tasks"],
        "task_index": str(data / "tasks.jsonl"),
        "grader_data": str(data / "grader.jsonl"),
        "index_sha256": manifest["index_sha256"],
        "grader_sha256": manifest["grader_sha256"],
        "policy_checkpoint": config["model_checkpoint"],
        "harness": snapshot(config["harness_path"], output / "harness"),
        "protocol": spec,
        "sandbox": "Orchard",
        "workers": 1,
        "runtime": config["runtime"],
        "serving": config["serving"],
        "seed": config["seed"],
        "prompt_specialization": "Pro V2 appends public Requirements/Interface; selected harness prompts unchanged",
        "image_digest": None,
        "limitation": "Image tags are not observed runtime digests; live validation pending",
    }
    write(output / "plan.json", plan)
    return plan


def execute(plan, config, output):
    output = Path(output)
    if sha(plan["task_index"]) != plan["index_sha256"] or sha(plan["grader_data"]) != plan["grader_sha256"]:
        raise ValueError("Frozen final dataset changed")
    if (output / "results.jsonl").exists():
        raise ValueError("Final output exists; use a new directory")
    spec = plan["protocol"]
    if "verifier_repo" in spec:
        os.environ["SWE_PRO_REPO"] = spec["verifier_repo"]
        os.environ["SWE_PRO_ENTRY_TIMEOUT"] = str(spec["verifier_seconds"])
        os.environ["SWE_PRO_CPU"] = str(spec["grader_cpu"])
        os.environ["SWE_PRO_MEM"] = spec["grader_memory"]
    verifier = importlib.import_module("minisweagent.run.benchmarks." + spec["module"])
    rows = [json.loads(line) for line in Path(plan["task_index"]).read_text().splitlines()][: plan["task_count"]]
    private = {r["instance_id"]: r for r in map(json.loads, Path(plan["grader_data"]).read_text().splitlines())}
    c = {**config["runtime"], "max_seconds": spec["seconds"]}
    env_config = {
        "base_url": os.environ[c["sandbox"]["endpoint_env"]],
        "api_key": os.environ[c["sandbox"]["key_env"]],
        "block_network": False,
        "sandbox_timeout": spec.get("verifier_seconds", 1800),
        "cpu": str(c["sandbox"]["cpu"]),
        "memory": c["sandbox"]["memory"],
        "forward_env": [],
    }
    with serve(plan["policy_checkpoint"], config, output / "serve") as client:
        for row in rows:
            iid = row["instance_id"]
            if row["split"] != "final_test":
                raise ValueError("Unexpected final data role")
            image = row["image_name"]
            if plan["benchmark"] != "pro-v2":
                from minisweagent.run.benchmarks.utils.images import get_swebench_docker_image_name

                image = get_swebench_docker_image_name(row)
            problem = row["problem_statement"]
            for key in ("requirements", "interface"):
                if row.get(key):
                    problem += f"\n\n## {key.title()}\n" + row[key]
            task = {
                "task_id": iid,
                "problem_statement": problem,
                "image": image,
                "cwd": spec["cwd"],
                "base_commit": row["base_commit"],
                "split": "final_test",
                "dataset_source": row["source"],
            }
            sampling_seed = int(digest([config["seed"], iid])[:8], 16) % 2147483647
            with TaskSandbox(task, c["sandbox"]) as sandbox:
                r = run_agent(
                    client,
                    sandbox,
                    task,
                    plan["harness"]["path"],
                    plan["policy_checkpoint"],
                    {**c, "sampling_seed": sampling_seed},
                    output / iid,
                )
            instance = private[iid]
            if plan["benchmark"] == "pro-v2":
                instance = verifier.map_row_to_instance(instance)
            else:
                instance = {**instance, "image_name": row["image_name"]}
            result = verifier.evaluate_instance_orchard(instance, r["patch"], env_config, logs_dir=output / "grading")
            infra = result.get("error") or result.get("exit_code", 0) in (-1, 124, 137)
            if plan["benchmark"] != "pro-v2":
                log = (output / "grading" / f"{iid}_log.txt").read_text()
                infra = infra or ">>>>> Start Test Output" not in log or ">>>>> End Test Output" not in log
            r.update(
                score=None if infra else float(result["resolved"]),
                reward=result,
                dataset_revision=plan["dataset_revision"],
                error_type="grading_infrastructure" if infra else None,
            )
            append(output / "results.jsonl", r)
            if infra:
                raise RuntimeError("Final grading infrastructure failed; score is null, see preserved report")


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["verified", "multilingual", "pro-v2"], required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/baseline_9b.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    a = parser.parse_args()
    config = load_config(a.config)
    plan = prepare(a.benchmark, config, a.output, a.limit)
    if a.execute:
        report = preflight(config, phase="final-eval")
        write(a.output / "preflight.json", report)
        if report["blockers"]:
            raise RuntimeError("Final preflight blocked: " + "; ".join(report["blockers"]))
        execute(plan, config, a.output)
    else:
        print(f"NOT_RUN: {plan['benchmark']} {plan['task_count']} tasks; plan={a.output / 'plan.json'}")


if __name__ == "__main__":
    main()
