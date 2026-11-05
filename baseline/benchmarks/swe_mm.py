"""SWE-bench Multimodal v2: existing mini agent + image inputs + official grading."""

import base64
import copy
import json
import os
import shlex
from pathlib import Path
from types import SimpleNamespace

import yaml

from baseline.records import ROOT, digest, sha, write

from .base import TaskInfo
from .official import OfficialBenchmark, OfficialTask, file_reference, path_value
from .swe_mm_assets import copy_public_observations, image_metadata
from .swe_mm_data import DATASET, REVISION, load_split


def runtime_settings(options):
    runtime = copy.deepcopy(options["runtime"])
    sandbox = runtime["sandbox"]
    path = path_value(sandbox["image_manifest"])
    sandbox["images"] = yaml.safe_load(path.read_text())["images"] if path.exists() else {}
    return runtime


class SWEMultimodal(OfficialBenchmark):
    name = "swe_mm"
    url = "https://github.com/SWE-bench/SWE-bench"
    commit = "87ab1f6ced28f75ba73ca899dc759b019310944a"

    def evaluation_role(self, config):
        split = config["official"]["split"]
        if split not in ("dev", "test"):
            raise ValueError("SWE-MM split must be dev or test")
        return "dev_eval" if split == "dev" else "final_test"

    def discover(self, config):
        options = config["official"]
        root, split = path_value(options["dataset_dir"]), options["split"]
        rows, manifest = load_split(root, split)
        return [
            OfficialTask(
                TaskInfo(row["task_id"], DATASET, split),
                row,
                {"dataset_dir": str(root), "split": split, "task_id": row["task_id"]},
                digest({"dataset_revision": REVISION, "split": manifest["splits"][split], "public": row}),
            )
            for row in rows
        ]

    def input_fingerprint(self, task):
        rows, manifest = load_split(task.native["dataset_dir"], task.info.split)
        public = next(r for r in rows if r["task_id"] == task.info.task_id)
        value = digest({"dataset_revision": REVISION, "split": manifest["splits"][task.info.split], "public": public})
        if value != task.fingerprint:
            raise ValueError("Frozen SWE-MM task changed")
        return value

    def plan(self, task, checkpoint, harness, config, output, seed):
        from baseline.harness import revision

        request = super().plan(task, checkpoint, harness, config, output, seed)
        request["agent_harness"] = {
            "path": str(path_value(config["official"]["agent_harness"])),
            "revision": revision(path_value(config["official"]["agent_harness"])),
        }
        request["runtime_hash"] = digest(
            {name: sha(ROOT / "baseline" / name) for name in ("runtime.py", "sandbox.py", "tencent.py")}
        )
        request["grader_patch_hash"] = sha(Path(__file__).with_name("swe_mm_transport.patch"))
        request["sandbox_images"] = runtime_settings(config["official"])["sandbox"]["images"]
        cache = config["official"].get("asset_dir")
        request["asset_manifests"] = (
            {kind: sha(path_value(cache) / kind / "manifest.json") for kind in ("public", "grader")} if cache else None
        )
        request["harness_revision"] = digest(
            {
                k: request[k]
                for k in ("harness_revision", "agent_harness", "runtime_hash", "grader_patch_hash", "sandbox_images", "asset_manifests")
            }
        )
        return request


def download_observations(public, output, asset_dir=None):
    """Only issue images, never patch/test images. Large bytes stay in artifact files."""
    import requests

    if asset_dir is not None:
        return copy_public_observations(asset_dir, public, output)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    observations = [{"kind": "text", "text": public["problem_statement"]}]
    for index, entry in enumerate(public["issue_images"]):
        url = entry if isinstance(entry, str) else entry["url"]
        if not url.startswith("https://"):
            raise ValueError("Issue images require HTTPS")
        chunks, size = [], 0
        with requests.get(url, timeout=60, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 20 * 1024 * 1024:
                    raise ValueError("Issue image exceeds 20 MiB; inspect it before increasing the budget")
                chunks.append(chunk)
        content = b"".join(chunks)
        metadata = image_metadata(content)
        path = output / f"issue_{index}.{metadata['format']}"
        path.write_bytes(content)
        reference = file_reference(path)
        reference["metadata"].update(url=url, width=metadata["width"], height=metadata["height"])
        observations.append({"kind": "image", "artifact": reference})
    write(output / "observations.json", observations)
    return observations


class ImageClient:
    """API boundary only: the existing mini agent owns all actions and control flow.

    Its normal text trace plus observations.json records the complete binding;
    data URIs are created only for transport and never written into JSON logs.
    """

    def __init__(self, client, observations):
        self.client = client
        self.images = [o["artifact"] for o in observations if o["kind"] == "image"]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        request = copy.deepcopy(request)
        first_user = next(m for m in request["messages"] if m["role"] == "user")
        content = first_user["content"]
        first_user["content"] = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
        for artifact in self.images:
            data = Path(artifact["path"]).read_bytes()
            if sha(artifact["path"]) != artifact["metadata"]["sha256"]:
                raise ValueError("Issue image changed after it was recorded")
            url = f"data:{artifact['media_type']};base64," + base64.b64encode(data).decode()
            first_user["content"].append({"type": "image_url", "image_url": {"url": url}})
        return self.client.chat.completions.create(**request)


def run_native(request):
    import httpx
    from openai import OpenAI

    from baseline.harness import revision, snapshot
    from baseline.runtime import run_agent
    from baseline.sandbox import TaskSandbox

    from .swe_mm_grading import grade, image_environment, with_environment

    options, native, output = request["options"], request["task"], Path(request["output"])
    if request["runtime_hash"] != digest(
        {name: sha(ROOT / "baseline" / name) for name in ("runtime.py", "sandbox.py", "tencent.py")}
    ):
        raise ValueError("SWE-MM runtime changed after planning")
    if request["grader_patch_hash"] != sha(Path(__file__).with_name("swe_mm_transport.patch")):
        raise ValueError("SWE-MM grader transport changed after planning")
    adapter = SWEMultimodal()
    config = {"official": {**options, "task_ids": [request["task_id"]]}}
    task_record = adapter.load_tasks(config, adapter.evaluation_role(config), 1)[0]
    if adapter.input_fingerprint(task_record) != request["task_fingerprint"]:
        raise ValueError("SWE-MM task fingerprint mismatch")
    if revision(request["agent_harness"]["path"]) != request["agent_harness"]["revision"]:
        raise ValueError("Agent harness changed after planning")
    harness = snapshot(request["agent_harness"]["path"], output / "agent_harness")
    public = task_record.public
    cache = path_value(options["asset_dir"]) if options.get("asset_dir") else None
    if cache and request["asset_manifests"] != {
        kind: sha(cache / kind / "manifest.json") for kind in ("public", "grader")
    }:
        raise ValueError("SWE-MM asset manifests changed after planning")
    observations = download_observations(public, output / "observations", cache)
    task = {
        "task_id": public["task_id"],
        "dataset_source": DATASET,
        "split": native["split"],
        "problem_statement": public["problem_statement"],
        "base_commit": public["base_commit"],
        "image": public["sandbox_image"],
        "cwd": "/testbed",
    }
    runtime = {**runtime_settings(options), "sampling_seed": request["seed"]}
    if runtime["sandbox"]["images"] != request["sandbox_images"]:
        raise ValueError("Task image manifest changed after planning")
    runtime["image_environment"] = image_environment(task, runtime, output)

    class JavaScriptSandbox(TaskSandbox):
        def raw(self, command, timeout=120, cwd=None):
            return super().raw(with_environment(command, runtime["image_environment"]), timeout, cwd=cwd)

        def prepare(self):
            # Read only executable paths, not the full image environment or credentials.
            node = (
                self.checked('. "${NVM_DIR:-/usr/local/nvm}/nvm.sh" && command -v node')["output"]
                .strip()
                .splitlines()[-1]
            )
            if not node.startswith("/") or node.startswith("/root/"):
                raise ValueError("Task Node executable is unavailable to the unprivileged agent")
            self.node_bin = str(Path(node).parent)
            super().prepare()
            if self.execute({"command": "node --version"})["returncode"]:
                raise RuntimeError("Node is not usable after sandbox isolation")

        def execute(self, action, cwd="", timeout=60):
            command = "export PATH=" + shlex.quote(self.node_bin) + ':"$PATH"; ' + action["command"]
            return super().execute({**action, "command": command}, cwd=cwd, timeout=timeout)

    with (
        OpenAI(
            api_key=os.environ[options["policy_api_key_env"]],
            base_url=options["policy_base_url"],
            max_retries=0,
            http_client=httpx.Client(trust_env=options.get("policy_trust_env", False)),
        ) as client,
        JavaScriptSandbox(task, runtime["sandbox"]) as sandbox,
    ):
        result = run_agent(
            ImageClient(client, observations),
            sandbox,
            task,
            harness["path"],
            request["model"],
            runtime,
            output / "native",
        )
    write(output / "agent_result.json", result)
    # Only the trusted verifier reads private rows, after the agent sandbox is killed.
    adapter.input_fingerprint(task_record)
    grader = Path(native["dataset_dir"]) / native["split"] / "grader.jsonl"
    row = next(r for r in map(json.loads, grader.read_text().splitlines()) if r["instance_id"] == task["task_id"])
    if cache:
        from .swe_mm_assets import grader_repo

        if sha(cache / "grader/manifest.json") != request["asset_manifests"]["grader"]:
            raise ValueError("Private asset manifest changed during execution")
        runtime["grader_task_repo"] = grader_repo(cache, row)
    prediction = {
        "instance_id": task["task_id"],
        "model_name_or_path": request["model"],
        "model_patch": result["patch"],
    }
    write(output / "prediction.json", prediction)
    report = grade(row, task, prediction, runtime, request["source"]["repo"], output)
    infra = (
        report.get("infra_failure")
        or report["missing_grader_assets"]
        or (
            not report["patch_successfully_applied"]
            and not report.get("empty_submission")
            and not report.get("patch_rejected")
        )
    )
    return {
        "task_id": task["task_id"],
        "seed_applied_to_model": True,
        "score": None if infra else float(report["resolved"]),
        "metrics": {
            "official_report": report,
            "dataset_revision": REVISION,
            "dataset_split": native["split"],
            "agent_harness": harness,
            "image_input": "issue_images_only",
            "seed_applied_to_model": True,
        },
        "error_type": "GradingInfrastructureError" if infra else None,
        "termination_reason": result["termination_reason"],
        "tokens": result["tokens"],
        "tool_calls": result["tool_calls"],
    }
