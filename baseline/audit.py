"""Read-only resource preflight. Missing resources are blockers, never simulated scores."""

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .checkpoint import inspect
from .data import tasks
from .records import ROOT, SLIME_COMMIT, sha
from .tencent import preflight as sandbox_preflight


def environment(config):
    packages = {}
    for name in (
        "torch",
        "ray",
        "sglang",
        "megatron-core",
        "transformer-engine",
        "transformers",
        "mini-swe-agent",
        "swebench",
        "e2b",
        "tensorboard",
        "straw-queue",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    repos = {}
    for name, path in {
        "baseline": ROOT,
        "slime": config["slime_root"],
    }.items():
        exists = (Path(path) / ".git").exists()
        repos[name] = {
            "commit": subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            if exists
            else None,
            "dirty": bool(subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True))
            if exists
            else None,
        }
    import minisweagent
    import torch

    gpu = [
        {"name": torch.cuda.get_device_name(i), "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())
    ]
    return {
        "python": platform.python_version(),
        "packages": packages,
        "mini_source": str(Path(minisweagent.__file__).resolve()),
        "repositories": repos,
        "cuda": torch.version.cuda,
        "nccl_available": torch.distributed.is_nccl_available(),
        "gpus": gpu,
        "tools": {name: shutil.which(name) for name in ("nvidia-smi", "nvcc", "docker", "kubectl", "claude")},
        "model": inspect(config["model_checkpoint"]),
        "split_manifest_sha256": sha(config["dataset_split_manifest"]),
    }


def preflight(config, *, phase="run", images=None):
    report = environment(config)
    blockers = []
    if phase != "oracle" and not report["gpus"]:
        blockers.append("No visible CUDA GPU in this allocation")
    if phase != "oracle" and report["packages"]["sglang"] is None:
        blockers.append("SGLang is not installed in this Conda environment")
    if phase in ("run", "p-step") and config["p_enabled"]:
        for name in ("ray", "megatron-core", "transformer-engine", "tensorboard", "straw-queue"):
            if report["packages"][name] is None:
                blockers.append(f"Training dependency missing: {name}")
        if len(report["gpus"]) < config["training"]["gpus"]:
            blockers.append("Allocation has fewer GPUs than the configured training budget")
    if images is None:
        selected = tasks(
            config["dataset_split_manifest"],
            "optimization_pool",
            max(config["h_task_sample_size"], config["p_task_sample_size"]),
        )
        if phase in ("run", "h-step", "cross-eval"):
            selected += tasks(config["dataset_split_manifest"], "dev_eval", config["dev_task_sample_size"])
        elif phase == "audit-holdout":
            selected = tasks(config["dataset_split_manifest"], "audit_holdout", config["dev_task_sample_size"])
        images = [r["metadata"]["image_url"] for r in selected]
    sandbox = sandbox_preflight(config["runtime"]["sandbox"], images)
    blockers.extend(sandbox["blockers"])
    if phase in ("run", "h-step") and config["h_enabled"] and config["proposer_mode"] == "api":
        for key in ("key_env", "model_env"):
            if not os.environ.get(config["proposer"][key]):
                blockers.append("Proposer environment variable missing: " + config["proposer"][key])
    if report["repositories"]["slime"]["commit"] != SLIME_COMMIT:
        blockers.append("Slime pin mismatch")
    if report["packages"]["swebench"] != "4.1.0":
        blockers.append("This baseline preserves Self-HPO's swebench==4.1.0 grading dependency")
    if report["packages"]["mini-swe-agent"] != "2.2.6+selfhpo" or not Path(report["mini_source"]).is_relative_to(
        ROOT / "mini-swe-agent/src"
    ):
        blockers.append("Install the pinned mini-SWE-agent editable from this worktree")
    return {
        **report,
        "sandbox": sandbox,
        "status": "BLOCKED" if blockers else "PREFLIGHT_ONLY",
        "blockers": blockers,
        "real_task_execution": "NOT_RUN",
        "gpu_training": "NOT_RUN",
    }
