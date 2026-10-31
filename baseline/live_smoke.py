"""Small, opt-in real API/Tencent probe. This is NOT a SWE or learning experiment.

Use the same seed AgentHarness, action parser, submission protocol, transport
commands and UID isolation. Only this probe accepts a general coding image;
the dataset path still requires exact task images and pinned registry digests.
"""

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import httpx
from e2b import Sandbox
from openai import OpenAI

from .config import load_config
from .data import tasks
from .execution import rollout
from .proposer import complete, parse_json
from .records import ROOT, write
from .runtime import run_agent
from .sandbox import TaskSandbox

SOURCE = "def clamp(value, low, high):\n    return min(low, max(value, high))\n"
CHECK = """import sys
sys.path.insert(0, '/testbed')
from ranges import clamp
cases = [(-3, 0, 10, 0), (5, 0, 10, 5), (20, 0, 10, 10),
         (-7, -5, -1, -5), (-3, -5, -1, -3), (2, 2, 2, 2)]
for value, low, high, expected in cases:
    assert clamp(value, low, high) == expected
print('SIX_PRIVATE_CHECKS_PASSED')
"""


@contextlib.contextmanager
def fixture_sandbox(settings, image, output, *, grader=False):
    """A synthetic fixture in a real sandbox; never a replacement SWE image."""
    task = {
        "task_id": "synthetic-clamp-smoke",
        "dataset_source": "synthetic connectivity fixture, not Self-HPO-RL-Data",
        "split": "connectivity_only",
        "image": image,
        "cwd": "/testbed",
        "base_commit": "HEAD",
        "problem_statement": (
            "Fix ranges.py: clamp(value, low, high) should return low when value is below low, "
            "high when above high, and value otherwise. Assume low <= high. Support negative numbers "
            "and equal bounds. Modify only ranges.py and submit its git patch."
        ),
    }
    box = TaskSandbox(task, settings, grader=grader)
    life = {"created": False, "cleanup": False, "image": image, "runtime_digest": None}
    try:
        options = {
            "api_key": os.environ[settings["key_env"]],
            "domain": os.environ[settings["domain_env"]],
            "request_timeout": 60,
            "validate_api_key": False,
            "retries": 0,
            "http_version": "1.1",
        }
        if settings["connection"] == "proxy":
            options["proxy"] = os.environ[settings["proxy_env"]]
        sbx = Sandbox.create(
            template=os.environ[settings["template_env"]],
            timeout=600,
            secure=True,
            allow_internet_access=False,
            metadata={
                "purpose": "rsi-coevolve-connectivity-only",
                "x-custom-config": json.dumps(
                    {
                        "image": image,
                        "imageRegistryType": "enterprise",
                        "command": ["/bin/sh", "-c"],
                        "args": ["/mnt/envd -port 49983"],
                        "resources": {"cpu": "1", "memory": "1Gi"},
                    }
                ),
            },
            **options,
        )
        # Attach the owned probe sandbox to the production command/UID adapter.
        # No patching of SDK methods or separate agent execution implementation.
        box._transport.sandbox = sbx
        box.sandbox = sbx
        box.execution_image, box.registry_digest = image, None
        life.update(created=True, sandbox_id=sbx.sandbox_id)
        write(output / "lifecycle.json", life)
        if settings["connection"] == "direct":
            sbx = Sandbox.connect(
                sbx.sandbox_id,
                sandbox_url=f"https://49983-{sbx.sandbox_id}.{options['domain']}",
                timeout=600,
                **options,
            )
            box._transport.sandbox = box.sandbox = sbx
        r = box.raw("mkdir -p /testbed", cwd="/tmp")
        if r["returncode"]:
            raise RuntimeError("Cannot create probe directory")
        box.upload("/testbed/ranges.py", SOURCE)
        box.checked(
            "git init -q; git add ranges.py; git -c user.name=probe -c user.email=probe@localhost commit -qm base"
        )
        box.prepare()
        if not grader:
            isolation = box.execute(
                {"command": 'id -u; test ! -r /root; test -z "${E2B_API_KEY:-}${LLM_CENTER_API_KEY:-}"'}
            )
            write(output / "isolation.json", isolation)
            if isolation["returncode"] or isolation["output"].strip() != "65534":
                raise RuntimeError("Probe UID, root protection or credential isolation failed")
        yield box, task
    finally:
        try:
            box.__exit__()
            life["cleanup"] = life["created"]
        finally:
            write(output / "lifecycle.json", life)


def run(config, output, *, image=None, policy_model=None, task_id=None):
    if image and task_id:
        raise ValueError("Use either a synthetic coding image or an optimization task")
    if bool(image or task_id) != bool(policy_model):
        raise ValueError("A coding probe requires an explicit API policy model")
    selected = []
    if task_id:
        selected = [
            r
            for r in tasks(config["dataset_split_manifest"], "optimization_pool")
            if r["metadata"]["instance_id"] == task_id
        ]
        if len(selected) != 1:
            raise ValueError("Task ID is not in optimization_pool")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory for each live smoke")
    output.mkdir(parents=True)
    report = {
        "experiment_kind": "real_connectivity_smoke",
        "simulated": False,
        "benchmark": False,
        "qwen_9b_validated": False,
        "policy_training": "NOT_RUN",
        "proposer_access_id": os.environ[config["proposer"]["model_env"]],
        "policy_access_id": policy_model,
        "task_id": task_id,
        "sandbox_image": image,
        "registry_digest": None,
        "image_reproducibility": "UNPINNED_PROBE_ONLY",
        "status": "RUNNING",
    }
    start = time.monotonic()
    try:
        content = complete(
            config["proposer"],
            {
                "messages": [{"role": "user", "content": 'API smoke: return only JSON {"ok":true}.'}],
                "max_tokens": 256,
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
            },
            output / "proposer",
        )
        if parse_json(content) != {"ok": True}:
            raise ValueError("Proposer did not return the requested JSON")
        report["proposer"] = "PASS"
        if task_id:
            # The normal task path retains all image pinning and original grader checks.
            runtime_config = {**config, "runtime": {**config["runtime"], "max_steps": 20, "max_seconds": 300}}
            with OpenAI(
                base_url=config["proposer"]["base_url"],
                api_key=os.environ[config["proposer"]["key_env"]],
                max_retries=0,
                http_client=httpx.Client(trust_env=config["proposer"]["trust_env"]),
            ) as client:
                result = rollout(
                    client,
                    selected[0],
                    policy_model,
                    config["harness_path"],
                    runtime_config,
                    output / "task",
                    config["seed"],
                )
            report.update(
                coding_task="EXECUTED",
                task_score=result["score"],
                tokens=result["tokens"],
                tool_calls=result["tool_calls"],
                sandbox_image=result["execution_image"],
                registry_digest=result["registry_digest_before_create"],
                image_reproducibility="REGISTRY_PINNED",
                tests_executed=result["reward"]["executed_tests"],
            )
            if not result["reward"]["executed_tests"]:
                raise RuntimeError("Model produced no gradable patch; end-to-end scoring is not verified")
        if image:
            settings = config["runtime"]["sandbox"]
            runtime = {**config["runtime"], "sampling_seed": config["seed"], "max_steps": 12, "max_seconds": 240}
            with (
                OpenAI(
                    base_url=config["proposer"]["base_url"],
                    api_key=os.environ[config["proposer"]["key_env"]],
                    max_retries=0,
                    http_client=httpx.Client(trust_env=config["proposer"]["trust_env"]),
                ) as client,
                fixture_sandbox(settings, image, output / "agent") as (box, task),
            ):
                result = run_agent(client, box, task, config["harness_path"], policy_model, runtime, output / "agent")
            write(output / "agent/result.json", result)
            patch = result["patch"]
            if not patch.strip():
                raise RuntimeError("Real policy did not submit a patch within the smoke budget")
            # Trusted checks are uploaded only into a fresh grader, after policy execution.
            with fixture_sandbox(settings, image, output / "grader", grader=True) as (box, _):
                box.upload("/root/private_check.py", CHECK)
                before = box.raw("python3 /root/private_check.py")
                box.upload("/tmp/probe.patch", patch)
                names = box.checked("git apply --numstat /tmp/probe.patch")["output"].strip().splitlines()
                if not names or any(line.split("\t")[-1] != "ranges.py" for line in names):
                    raise ValueError("Probe patch modifies files outside the allowed source")
                box.checked("git apply --check /tmp/probe.patch && git apply /tmp/probe.patch")
                after = box.raw("python3 /root/private_check.py")
                write(output / "grader/checks.json", {"before": before, "after": after})
                if (
                    before["returncode"] != 1
                    or after["returncode"] != 0
                    or "SIX_PRIVATE_CHECKS_PASSED" not in after["output"]
                ):
                    raise RuntimeError("Private fixture checks did not show a verified repair")
            report.update(
                coding_task="PASS", private_checks=6, tokens=result["tokens"], tool_calls=result["tool_calls"]
            )
        report["status"] = "PASS"
        return report
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__)
        raise
    finally:
        report["latency_seconds"] = time.monotonic() - start
        write(output / "report.json", report)


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=ROOT / "configs/baseline_9b.yaml")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image", help="General Tencent coding image; probe only, never a SWE task substitution")
    p.add_argument("--task-id", help="One optimization task; requires its pinned Tencent image mapping")
    p.add_argument("--policy-model", help="Explicit API policy accessId, solely for this no-GPU probe")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.image and args.task_id:
        p.error("--image and --task-id are mutually exclusive")
    if bool(args.image or args.task_id) != bool(args.policy_model):
        p.error("Coding probe requires --policy-model and either --image or --task-id")
    if not args.execute:
        p.error("--execute required: this probe uses real API/sandbox resources")
    print(
        json.dumps(
            run(
                load_config(args.config),
                args.output,
                image=args.image,
                policy_model=args.policy_model,
                task_id=args.task_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
