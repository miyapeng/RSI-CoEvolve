"""Offline discovery/preflight and explicit official evaluation; never starts H/P training."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from baseline.records import write

from . import get_benchmark
from .official import path_value


def preflight(config):
    """Read-only checks: no images pulled, sandboxes created or model calls made."""
    adapter = get_benchmark(config)
    report = {"benchmark": adapter.name, "checks": {}, "live_execution": "NOT_RUN"}
    checks = report["checks"]
    try:
        report["source"] = adapter.source(config)
        checks["source"] = True
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        checks["source"] = False
        report["source_error"] = str(exc)
    try:
        rows = adapter.load_tasks(config, "final_test", 1)
        checks["local_task"] = True
        report["example_task_id"] = rows[0].info.task_id
    except (ValueError, OSError, KeyError) as exc:
        checks["local_task"] = False
        report["task_error"] = str(exc)
    options = config["official"]
    native_modules = {
        "vision2web": ["vision2web.inference.engine", "vision2web.evaluation.engine"],
        "osworld_verified": ["desktop_env.desktop_env", "mm_agents.agent", "lib_run_single"],
        "arc_agi_3": ["benchmarking.swarm"],
    }[adapter.name]
    # Probe a separate process so an editable package from another project cannot shadow the pinned checkout.
    probe = (
        "import sys,importlib,json; sys.path.insert(0,sys.argv[1]); "
        "print(json.dumps({n:importlib.import_module(n).__file__ for n in sys.argv[2:]}))"
    )
    try:
        p = subprocess.run(
            [options.get("python", "python"), "-c", probe, str(path_value(options["repo"])), *native_modules],
            capture_output=True,
            text=True,
            timeout=45,
        )
        checks["native_imports"] = p.returncode == 0
        if p.returncode == 0:
            report["native_modules"] = json.loads(p.stdout.strip().splitlines()[-1])
        else:
            # Do not copy arbitrary SDK stderr (which can contain credentials) to a public report.
            report["import_error"] = (
                "Native imports failed; install the pinned upstream dependencies in its Conda environment"
            )
    except (OSError, subprocess.TimeoutExpired):
        checks["native_imports"] = False
    if adapter.name == "vision2web":
        for field in ("policy_api_key_env", "functional_api_key_env", "visual_api_key_env"):
            checks[field] = bool(os.environ.get(options[field]))
        transport = options.get("transport", "docker")
        checks["transport"] = transport in {"docker", "tencent_e2b"}
        if transport == "tencent_e2b":
            from baseline.tencent import SDK_VERSION

            from .vision2web_e2b import validate

            try:
                cfg = validate(options)
                checks["tencent_config"] = True
                checks["e2b_key"] = bool(os.environ.get(cfg["key_env"]))
                if cfg["connection"] == "proxy":
                    checks["e2b_proxy"] = bool(os.environ.get(cfg["proxy_env"]))
                p = subprocess.run(
                    [
                        options.get("python", "python"),
                        "-c",
                        "import importlib.metadata; print(importlib.metadata.version('e2b'))",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                checks["e2b_sdk"] = p.returncode == 0 and p.stdout.strip() == SDK_VERSION
                report["remote_image_tools"] = "NOT_VERIFIED; run the Tencent probe"
            except (ValueError, KeyError, OSError) as exc:
                checks["tencent_config"] = False
                report["tencent_error_type"] = type(exc).__name__
                report["tencent_requirement"] = (
                    "Set VISION2WEB_SANDBOX_IMAGE to a real TCR tag@sha256 and check the tencent settings"
                )
    elif adapter.name == "osworld_verified":
        checks["provider"] = options["provider_name"] in {"docker", "vmware", "virtualbox", "tencent_e2b"}
        checks["policy_api_key_env"] = bool(os.environ.get(options["policy_api_key_env"]))
        if options["provider_name"] == "tencent_e2b":
            from .osworld_tencent import source

            checks["sandbox_key"] = bool(os.environ.get(options["tencent"]["key_env"]))
            checks["patch_tool"] = bool(shutil.which("patch"))
            try:
                probe = subprocess.run(
                    [
                        options.get("python", "python"),
                        "-c",
                        "import e2b,aiohttp; import importlib.metadata as m; print(m.version('e2b'))",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                checks["e2b_sdk"] = probe.returncode == 0 and probe.stdout.strip() == "2.62.0"
            except (OSError, subprocess.TimeoutExpired):
                checks["e2b_sdk"] = False
            try:
                report["tencent_source"] = source(options)
                checks["tencent_source"] = True
            except (ValueError, OSError, subprocess.CalledProcessError):
                checks["tencent_source"] = False
            report["remote_desktop"] = "NOT_VERIFIED by offline preflight"
        else:
            checks["local_vm"] = bool(options.get("path_to_vm")) and path_value(options["path_to_vm"]).is_file()
            if options["provider_name"] == "docker":
                checks["kvm"] = os.access("/dev/kvm", os.R_OK | os.W_OK)
    else:
        checks["arc_api_key"] = bool(os.environ.get("ARC_API_KEY"))
        from .arc_agi_3 import model_profile

        try:
            entry = model_profile(options)
        except (ValueError, OSError, KeyError):
            entry = None
        checks["model_profile"] = entry is not None and entry["request"]["model"] == config["model"]
        if entry:
            report["native_profile"] = entry  # Credential environment names only, never their values.
            checks["policy_api_key_env"] = bool(os.environ.get(entry["client"]["api_key_env"]))
    if (adapter.name == "vision2web" and options.get("transport", "docker") == "docker") or (
        adapter.name == "osworld_verified" and options["provider_name"] == "docker"
    ):
        checks["docker_image"] = False
        if shutil.which("docker"):
            image = options.get("sandbox_image", "happysixd/osworld-docker")
            p = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=20)
            checks["docker_image"] = p.returncode == 0
    report["ready_for_live_attempt"] = all(checks.values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "plan", "preflight", "run"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--python", help="Python executable in the benchmark's Conda environment")
    parser.add_argument("--dataset-dir", help="Existing Vision2Web data; no downloads")
    parser.add_argument("--output", help="New run directory; also optional report path for preflight")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.python:
        config["official"]["python"] = args.python
    if args.dataset_dir:
        config["official"]["dataset_dir"] = str(Path(args.dataset_dir).resolve())
    if args.task_id:
        config["official"]["task_ids"] = args.task_id
    adapter = get_benchmark(config)
    if args.action == "preflight":
        report = preflight(config)
        if args.output:
            write(Path(args.output) / "preflight.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["ready_for_live_attempt"] else 2
    adapter.source(config)
    rows = adapter.load_tasks(config, "final_test", args.limit)
    if args.action == "list":
        print(json.dumps([{"task_id": r.info.task_id, "split": r.info.split} for r in rows], indent=2))
        return 0
    if not args.output:
        parser.error("--output is required for plan/run")
    root = Path(args.output).resolve()
    if args.action == "plan":
        plans = [
            adapter.plan(r, config["model"], config["official"]["repo"], config, root / r.info.task_id, config["seed"])
            for r in rows
        ]
        print(json.dumps(plans, indent=2))
        return 0
    report = preflight(config)
    if not report["ready_for_live_attempt"]:
        print(json.dumps(report, indent=2))
        return 2
    root.mkdir(parents=True, exist_ok=False)
    write(root / "config.resolved.json", config)
    write(root / "preflight.json", report)
    # This evaluation path needs no SWE SGLang server or SWE harness snapshot.
    from baseline.execution import evaluate

    results = evaluate(None, rows, config["model"], config["official"]["repo"], config, root / "episodes")
    complete = all(not r.get("error_type") for r in results)
    write(
        root / "summary.json",
        {"benchmark": adapter.name, "results": results, "complete": complete, "training": "NOT_RUN"},
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
