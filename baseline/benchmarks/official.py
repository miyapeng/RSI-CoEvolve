"""Run pinned official harnesses in their own Python environments, one task at a time."""

import math
import mimetypes
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from baseline.records import ROOT, digest, read, sha, write

from .base import TaskInfo


@dataclass(frozen=True)
class OfficialTask:
    info: TaskInfo
    public: dict
    native: dict  # Host-only paths/IDs; never a model prompt.
    fingerprint: str


def file_reference(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        "metadata": {"sha256": sha(path)},
    }


def path_value(value):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in value:
        raise ValueError(f"Set the environment variable used in path: {value}")
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def safe_id(value):
    # IDs also become output paths. No shell interpretation is involved.
    if not value or any(p in ("", ".", "..") for p in value.split("/")) or "\\" in value:
        raise ValueError(f"Invalid task/model identifier: {value!r}")
    return value


class OfficialBenchmark:
    """Small subprocess boundary; subclasses retain their native runtime and verifier."""

    def task_info(self, task):
        return task.info

    def public_task(self, task):
        return task.public

    def input_fingerprint(self, task):
        return task.fingerprint

    def source(self, config):
        repo = path_value(config["official"]["repo"])
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
        if commit != self.commit or dirty:
            raise ValueError(
                f"{self.name} requires clean tracked source at {self.commit}; found {commit}, dirty={bool(dirty)}"
            )
        return {"url": self.url, "commit": commit, "repo": str(repo)}

    def load_tasks(self, config, role, limit):
        if role != "final_test":
            raise ValueError(f"{self.name} is evaluation-only; do not use final benchmark tasks for H/P selection")
        if limit < 1:
            raise ValueError("Task limit must be positive")
        rows = self.discover(config)
        by_id = {r.info.task_id: r for r in rows}
        if len(by_id) != len(rows):
            raise ValueError("Duplicate official task IDs")
        selected = config["official"].get("task_ids", [])
        if selected:
            if len(set(selected)) != len(selected):
                raise ValueError("Duplicate requested task IDs")
            rows = [by_id[safe_id(i)] for i in selected]  # Missing requested tasks fail explicitly.
        if not rows:
            raise ValueError("No tasks found; supply local data / explicit official task IDs")
        return rows[:limit]

    def plan(self, task, checkpoint, harness, config, output, seed):
        source = self.source(config)
        if path_value(harness) != Path(source["repo"]):
            raise ValueError("Official harness path must match the pinned repository")
        if checkpoint != config["model"]:
            raise ValueError("Policy identity must match the model passed to the official harness")
        options = dict(config["official"])
        if self.name == "vision2web" and options.get("transport", "docker") == "tencent_e2b":
            from .vision2web_e2b import validate

            validate(options)
        if options.get("path_to_vm"):
            options["path_to_vm"] = str(path_value(options["path_to_vm"]))
        # Credentials are environment references only, never request JSON or argv.
        for key in options:
            if key in {"api_key", "token", "secret", "password"}:
                raise ValueError("Use credential environment variable names, not literal secrets")
        request = {
            "benchmark": self.name,
            "task_id": task.info.task_id,
            "task": task.native,
            "model": checkpoint,
            "options": options,
            "source": source,
            "seed": seed,
            "output": str(Path(output).resolve()),
            "task_fingerprint": self.input_fingerprint(task),
        }
        request["adapter_code_hash"] = digest({p.name: sha(p) for p in Path(__file__).parent.glob("*.py")})
        request["harness_revision"] = digest({"source": source["commit"], "options": options})
        return request

    def verify(self, task, episode, runtime_config, output):
        """Normalize an official verifier's output, without re-scoring it."""
        native = read(episode["native_result"])
        if native["task_id"] != task.info.task_id:
            raise ValueError("Official result task identity mismatch")
        score = native["score"]
        if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError("Invalid official score")
        verified = {"score": score, "training_reward": None, "details": native["metrics"]}
        write(Path(output) / "verification.json", verified)
        return verified

    def run_episode(self, client, task, checkpoint, harness, config, output, seed):
        if client is not None:
            raise ValueError("Official harnesses own their model client; pass client=None")
        output = Path(output).resolve()
        request = self.plan(task, checkpoint, harness, config, output, seed)
        # A new directory prevents native 'already completed' logic from reusing stale scores.
        output.mkdir(parents=True, exist_ok=False)
        write(output / "request.json", request)
        write(output / "public_task.json", self.public_task(task))
        command = [
            config["official"].get("python", "python"),
            str(Path(__file__).with_name("official_worker.py")),
            str(output / "request.json"),
        ]
        started = time.monotonic()
        try:
            with (output / "official.log").open("w") as log:
                os.chmod(log.name, 0o600)
                subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
            verified = self.verify(task, {"native_result": output / "native_result.json"}, {}, output)
        except Exception as exc:
            write(
                output / "failure.json",
                {
                    "task_id": task.info.task_id,
                    "policy_checkpoint": checkpoint,
                    "harness_revision": request["harness_revision"],
                    "score": None,
                    "error_type": type(exc).__name__,
                    "log": str(output / "official.log"),
                    "simulated": False,
                },
            )
            raise
        native = read(output / "native_result.json")
        artifacts = [
            file_reference(p)
            for p in output.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp4"}
        ]
        return {
            "task_id": task.info.task_id,
            "dataset_source": task.info.dataset_source,
            "split": task.info.split,
            "policy_checkpoint": checkpoint,
            "harness_revision": request["harness_revision"],
            "sampling_seed": seed,
            "sampling_params": {"seed_applied_to_model": False, "official_options": config["official"]},
            "trajectory": str(output / "native"),
            "termination_reason": native["termination_reason"],
            "error_type": native.get("error_type"),
            "score": verified["score"],
            "training_reward": None,
            "reward": verified["details"],
            "tokens": native.get("tokens"),
            "tool_calls": native.get("tool_calls"),
            "latency_seconds": time.monotonic() - started,
            "simulated": native.get("simulated", False),
            "artifacts": artifacts,
            "dataset_fingerprint": request["task_fingerprint"],
            "official_source": request["source"],
            "adapter_code_hash": request["adapter_code_hash"],
        }
