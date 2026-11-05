"""Vision2Web official inference, browser verifier and ResultAnalyzer."""

import json
from pathlib import Path

from baseline.records import digest, sha

from .base import TaskInfo
from .official import OfficialBenchmark, OfficialTask, file_reference, path_value, safe_id


class Vision2Web(OfficialBenchmark):
    name = "vision2web"
    url = "https://github.com/zai-org/Vision2Web"
    commit = "d80a4a3d0ea7326541ad87314294269a633789e4"

    def input_fingerprint(self, task):
        root = Path(task.native["dataset_dir"]) / task.info.task_id
        return digest({str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()})

    def discover(self, config):
        root = path_value(config["official"]["dataset_dir"])
        if not root.is_dir():
            raise FileNotFoundError(root)
        rows = []
        for kind in ("webpage", "frontend", "website"):
            for project in sorted((root / kind).glob("*")):
                if not project.is_dir() or project.name.startswith("."):
                    continue
                iid = safe_id(f"{kind}/{project.name}")
                workflow = project / "workflow.json"
                if not workflow.is_file() or not (project / "prototypes").is_dir():
                    raise ValueError(f"Incomplete official project: {iid}")
                # This is a public projection, not a reimplementation of the official prompt.
                observations = []
                requirements = project / ("prompt.txt" if kind == "frontend" else "prd.md")
                if kind != "webpage":
                    observations.append({"kind": "text", "text": requirements.read_text()})
                for image in sorted((project / "prototypes").iterdir()):
                    if image.suffix.lower() in {".jpg", ".png", ".jpeg", ".webp"}:
                        observations.append({"kind": "image", "artifact": file_reference(image)})
                public = {"task_id": iid, "task_type": kind, "observations": observations}
                if (project / "resources").is_dir():
                    public["resources_dir"] = str(project / "resources")
                rows.append(
                    OfficialTask(
                        TaskInfo(iid, "zai-org/Vision2Web", "final_test"),
                        public,
                        {"dataset_dir": str(root), "name": project.name, "task_type": kind},
                        digest({"public": public, "workflow_sha256": sha(workflow)}),
                    )
                )
        return rows


def evaluation_artifacts(workspace, project):
    """Report missing verifier evidence separately from the native zero-filled scores."""
    workflows = json.loads((project.path / "workflow.json").read_text())
    expected = []
    for idx, workflow in enumerate(workflows):
        for name in workflow.get("prototype", {}):
            expected.append(workspace / "test_results/prototypes" / f"{name}_scores.json")
        for case_idx, _ in enumerate(workflow.get("content", [])):
            expected.append(workspace / f"test_results/workflow_{idx}/test_case_{case_idx}/result.json")
    missing, invalid = [], []
    for path in expected:
        relative = str(path.relative_to(workspace))
        if not path.is_file():
            missing.append(relative)
            continue
        try:
            value = json.loads(path.read_text())
            valid = (
                (isinstance(value, list) and bool(value) and all("score" in x for x in value))
                if path.name.endswith("_scores.json")
                else (isinstance(value, dict) and bool(value.get("result")))
            )
            if not valid:
                invalid.append(relative)
        except (ValueError, TypeError):
            invalid.append(relative)
    return {
        "expected_files": len(expected),
        "missing": missing,
        "invalid": invalid,
        "complete": bool(expected) and not missing and not invalid,
    }


async def run_native(request):
    """Called only in the official environment. No second coding agent loop."""
    import os

    from vision2web.analyze.analyzer import ResultAnalyzer
    from vision2web.core.config import Config
    from vision2web.core.dataset import DatasetManager
    from vision2web.evaluation.engine import EvaluationEngine
    from vision2web.inference.engine import InferenceEngine

    from baseline.records import write

    options, task = request["options"], request["task"]
    c = Config()
    c.datasets_dir = Path(task["dataset_dir"])
    c.results_dir = Path(request["output"]) / "native"
    c.sandbox.image = options["sandbox_image"]
    c.inference.framework = options["framework"]
    c.inference.model = request["model"]
    # Native analyzer encodes model as one directory/colon-separated key.
    if any(x in c.inference.model for x in ("/", ":")):
        raise ValueError("Vision2Web native analyzer requires a model alias without '/' or ':'")
    c.inference.api_key = os.environ[options["policy_api_key_env"]]
    c.inference.base_url = options.get("policy_base_url")
    c.inference.timeout = options["timeout"]
    c.inference.max_workers = 1
    c.evaluation.functional_model = options["functional_model"]
    c.evaluation.functional_api_key = os.environ[options["functional_api_key_env"]]
    c.evaluation.functional_base_url = options.get("functional_base_url")
    c.evaluation.visual_model = options["visual_model"]
    c.evaluation.api_key = os.environ[options["visual_api_key_env"]]
    c.evaluation.base_url = options.get("visual_base_url")
    c.evaluation.max_workers = 1
    project = DatasetManager(c.datasets_dir).get_project(task["name"], task["task_type"])
    if project is None:
        raise ValueError("Official DatasetManager could not load the selected task")
    generated = await InferenceEngine(c).run_single_project(project)
    write(Path(request["output"]) / "inference.json", generated)
    if generated["status"] != "success":
        raise RuntimeError("Official inference did not complete; see inference.json")
    workspace = c.results_dir / project.task_type / c.inference.framework / c.inference.model / project.name
    evaluated = await EvaluationEngine(c).evaluate_single_project(
        {
            "name": project.name,
            "task_type": project.task_type,
            "workspace": workspace,
            "framework": c.inference.framework,
            "model": c.inference.model,
        },
        project,
    )
    analysis = ResultAnalyzer(str(c.results_dir)).analyze(str(c.datasets_dir))
    write(Path(request["output"]) / "analysis.json", analysis)
    artifacts = evaluation_artifacts(workspace, project)
    write(Path(request["output"]) / "evaluation_artifacts.json", artifacts)
    complete = evaluated["status"] == "success" and artifacts["complete"]
    # Keep official VS/FS and failure status. No invented scalar reward or weighting.
    return {
        "task_id": request["task_id"],
        "score": None,
        "metrics": {"analysis": analysis, "evaluation": evaluated, "artifacts": artifacts},
        "termination_reason": "official_evaluation_" + (evaluated["status"] if complete else "incomplete"),
        "error_type": None if complete else "OfficialEvaluationIncomplete",
    }
