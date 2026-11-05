"""Explicit Tencent transport for a per-run copy of the official SWE-bench grader.

No patch application, test selection, log parsing or reward formula is reimplemented.
"""

import importlib.util
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from baseline.records import sha, write
from baseline.sandbox import TaskSandbox


def image_environment(task, runtime, output):
    """Restore the pinned OCI image's ENV, which envd does not inherit on AGS.

    Read only image configuration, never os.environ or the host's credentials.
    """
    mapping = runtime["sandbox"]["images"][task["image"]]
    uri = mapping["image"].rsplit(":", 1)[0] + "@" + mapping["digest"]
    result = subprocess.run(
        ["skopeo", "inspect", "--config", "docker://" + uri], capture_output=True, text=True, check=True, timeout=120
    )
    config = json.loads(result.stdout)
    env = dict(entry.split("=", 1) for entry in config["config"].get("Env", []))
    write(Path(output) / "image_environment.json", {"image": uri, "environment": env})
    return env


def with_environment(command, environment):
    return shlex.join(["env", *(f"{k}={v}" for k, v in environment.items()), "bash", "-c", command])


def load_grader(repo, output):
    directory = Path(output) / "official_grader"
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "run_evaluation.py"
    original = Path(repo) / "swebench/harness/run_evaluation.py"
    shutil.copyfile(original, target)
    patch = Path(__file__).with_name("swe_mm_transport.patch")
    subprocess.run(["patch", "--batch", "--fuzz=0", str(target), str(patch)], check=True, capture_output=True)
    write(
        directory / "provenance.json",
        {"original_sha256": sha(original), "patch_sha256": sha(patch), "adapted_sha256": sha(target)},
    )
    spec = importlib.util.spec_from_file_location("swe_mm_official_grader", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GradingTransport:
    def __init__(self, task, config, environment):
        self.task, self.config = task, config
        self.image_env = environment
        self.environment = None
        self.patch_rejected = False

    def create_container(self, spec, client, run_id, logger):
        if spec.image != self.task["image"]:
            raise ValueError("Official grader task image mismatch")
        self.environment = TaskSandbox(self.task, self.config, grader=True).__enter__()
        # The few Docker methods used by the original verifier, explicitly injected.
        self.id = self.environment.sandbox.sandbox_id
        return self

    def start(self):
        pass  # E2B creation has already started the sandbox.

    def exec_run(self, command, workdir=None, user="root"):
        if user != "root":
            raise ValueError("Unexpected official verifier user")
        command = shlex.join(command) if isinstance(command, list) else command
        result = self.environment.raw(with_environment(command, self.image_env), cwd=workdir)
        # Native run_instance performs this exact final check only after every
        # patch application attempt failed. That is a model submission failure,
        # distinct from SDK/transport failure (which raises before this point).
        if command == "git apply --check --reverse /tmp/patch.diff":
            self.patch_rejected = result["returncode"] != 0
        return SimpleNamespace(exit_code=result["returncode"], output=result["output"].encode())

    def copy_file(self, container, source, destination):
        self.environment._transport.sandbox.files.write(str(destination), Path(source).read_bytes(), user="root")

    def execute(self, container, command, timeout):
        start = time.monotonic()
        result = self.environment.raw(with_environment(command, self.image_env), timeout=timeout)
        return result["output"], result["returncode"] in (124, 137), time.monotonic() - start

    def cleanup(self, client, container, logger):
        if self.environment is not None:
            self.environment.__exit__()
            self.environment = None


def grade(row, task, prediction, runtime, repo, output):
    from swebench.harness.grading import get_eval_report
    from swebench.harness.utils import make_test_spec

    module = load_grader(repo, output)
    if not prediction["model_patch"]:
        report = get_eval_report(make_test_spec(row), {**prediction, "model_patch": None}, "", True)[task["task_id"]]
        return {**report, "missing_grader_assets": [], "logs": None, "empty_submission": True}
    env = runtime.get("image_environment")
    if env is None:
        env = image_environment(task, runtime, output)
    transport = GradingTransport(task, runtime["sandbox"], env)
    # Absolute run_id anchors native logs under this episode, independent of cwd.
    result = module.run_instance(
        make_test_spec(row),
        prediction,
        None,
        str(Path(output).resolve() / "grading"),
        timeout=runtime["grading_timeout"],
        transport=transport,
        task_repo=runtime.get("grader_task_repo"),
    )
    model = prediction["model_name_or_path"].replace("/", "__")
    logs = Path(output) / "grading" / model / task["task_id"]
    if result is None:
        if transport.patch_rejected:
            report = get_eval_report(make_test_spec(row), prediction, str(logs / "run_instance.log"), True)[
                task["task_id"]
            ]
            return {**report, "patch_rejected": True, "missing_grader_assets": [], "logs": str(logs)}
        raise RuntimeError("Official SWE-MM grader did not produce a report; inspect grading/**/run_instance.log")
    report = result[1][task["task_id"]]
    # Missing binary references must not masquerade as a valid negative result.
    expected = [
        a for k in ("patch", "test_patch") for a in make_test_spec(row).image_assets.get(k, []) if a.get("path")
    ]
    missing = [a["path"] for a in expected if not (logs / "image_assets" / a["path"].replace("/", "__")).is_file()]
    report["missing_grader_assets"] = missing
    report["logs"] = str(logs)
    return report
