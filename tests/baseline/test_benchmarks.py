"""CPU interface fixtures only: no multimodal inference, RL, or remote execution."""

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_baseline import completion

from baseline import benchmarks, execution, outer_loop, reward, runtime, sandbox
from baseline.benchmarks.base import ArtifactReference, EpisodeResult, Observation, TaskInfo, Verification
from baseline.config import load_config
from baseline.data import public_task, tasks
from baseline.harness import revision
from baseline.records import ROOT, read, sha, write


@dataclass
class FixtureTask:
    task_id: str
    split: str
    screen: str
    private: dict


class FixtureBenchmark:
    name = "fixture"

    def load_tasks(self, config, role, limit):
        return [FixtureTask(**row) for row in read(config["dataset_split_manifest"])["tasks"] if row["split"] == role][
            :limit
        ]

    def task_info(self, task):
        return TaskInfo(task.task_id, "CPU interface fixture", task.split)

    def public_task(self, task):
        artifact: ArtifactReference = {
            "path": task.screen,
            "media_type": "image/svg+xml",
            "metadata": {"width": 16, "height": 16, "sha256": sha(task.screen)},
        }
        observations: list[Observation] = [
            {"kind": "text", "text": "Return the word blue."},
            {"kind": "image", "artifact": artifact},
            {"kind": "state", "data": {"turn": 0}},
            {"kind": "tool_result", "data": {"name": "fixture", "status": "ready"}},
        ]
        return {"task_id": task.task_id, "observations": observations}

    def verify(self, task, episode, runtime_config, output) -> Verification:
        passed = episode["answer"] == task.private["expected"]
        verified = {
            "score": float(passed),
            "training_reward": None,
            "details": {"passed": passed, "test_fixture": True},
        }
        write(Path(output) / "verification.json", verified)
        return verified

    def run_episode(self, client, task, checkpoint, harness, config, output, seed) -> EpisodeResult:
        visible = self.public_task(task)
        answer = client(visible)  # A separate deterministic runtime; never receives task.private.
        trajectory = Path(output) / "trajectory.json"
        write(trajectory, {"observations": visible, "answer": answer, "test_fixture": True})
        result = {
            "task_id": task.task_id,
            "dataset_source": self.task_info(task).dataset_source,
            "split": task.split,
            "policy_checkpoint": checkpoint,
            "harness_revision": revision(harness),
            "sampling_seed": seed,
            "sampling_params": {},
            "trajectory": str(trajectory),
            "termination_reason": "fixture_completed",
            "error_type": None,
            "tokens": None,
            "tool_calls": None,
            "latency_seconds": None,
            "simulated": True,
            "answer": answer,
            "artifacts": [visible["observations"][1]["artifact"]],
        }
        verified = self.verify(task, result, config.get("runtime", {}), Path(output) / "grading")
        result.update(score=verified["score"], training_reward=verified["training_reward"], reward=verified["details"])
        return result


@pytest.fixture
def fixture_config(monkeypatch, tmp_path):
    monkeypatch.setitem(benchmarks.ADAPTERS, "fixture", FixtureBenchmark)
    screen = tmp_path / "screen.svg"
    screen.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect width="16" height="16" fill="blue"/></svg>'
    )
    manifest = tmp_path / "manifest.json"
    write(
        manifest,
        {
            "tasks": [
                {
                    "task_id": role,
                    "split": role,
                    "screen": str(screen),
                    "private": {"expected": "blue", "grader_secret": "HIDDEN_EVALUATOR_SENTINEL"},
                }
                for role in ("optimization_pool", "dev_eval", "audit_holdout")
            ]
        },
    )
    config = load_config(ROOT / "configs/baseline_9b.yaml")
    config.update(
        benchmark="fixture",
        dataset_split_manifest=str(manifest),
        h_task_sample_size=1,
        p_task_sample_size=1,
        dev_task_sample_size=1,
        h_enabled=False,
        p_enabled=False,
        n_outer_rounds=2,
    )
    return config


def fixture_agent(visible):
    assert set(visible) == {"task_id", "observations"}
    assert "HIDDEN_EVALUATOR_SENTINEL" not in json.dumps(visible)
    return "blue"  # Intentionally does not infer anything from the image.


def test_second_adapter_observations_verification_and_outer_loop(fixture_config, tmp_path):
    c = fixture_config
    adapter = benchmarks.get_benchmark(c)
    rows = benchmarks.configured_tasks(c, "dev_eval", 1)
    results = execution.evaluate(
        fixture_agent, rows, "fixture-policy", c["harness_path"], {**c, "eval_repetitions": 2}, tmp_path / "episodes"
    )
    assert [r["score"] for r in results] == [1.0, 1.0]
    assert results[0]["sampling_seed"] != results[1]["sampling_seed"]
    assert all(r["simulated"] and r["tokens"] is None and r["benchmark"] == "fixture" for r in results)
    for result in results:
        artifact = result["artifacts"][0]
        assert Path(artifact["path"]).is_file() and artifact["media_type"].startswith("image/")
        assert "sandbox_image" not in artifact and "base64" not in json.dumps(result)
        assert read(Path(result["trajectory"]).parent / "result.json")["benchmark"] == "fixture"
    assert adapter.verify(rows[0], {"answer": "wrong"}, {}, tmp_path / "wrong")["score"] == 0.0
    for file in (tmp_path / "episodes").rglob("*.json*"):
        assert "HIDDEN_EVALUATOR_SENTINEL" not in file.read_text()

    class Ops(outer_loop.Operations):
        @staticmethod
        @contextlib.contextmanager
        def serve(*args):
            yield fixture_agent

    # Same orchestrator and real evaluate(), only the serving client is a fixture.
    state = outer_loop.run(c, tmp_path / "base", operations=Ops(), test_only=True)
    assert state["next_round"] == 2 and state["test_only"]
    for rd in (tmp_path / "base/rounds").iterdir():
        for cell in read(rd / "eval_pairs.json").values():
            assert cell[0]["benchmark"] == "fixture" and cell[0]["score"] == 1
    assert outer_loop.run(c, tmp_path / "base", resume=True, operations=Ops(), test_only=True) == state


@pytest.mark.parametrize("resolved", [True, False, "infra_failure"])
def test_swe_delegates_to_same_runtime_and_private_grader(monkeypatch, tmp_path, resolved):
    c = load_config(ROOT / "configs/baseline_9b.yaml")
    adapter = benchmarks.get_benchmark(c)
    assert adapter.name == "swe" and benchmarks.get_benchmark({"benchmark": "swe"}).name == "swe"
    row = tasks(c["dataset_split_manifest"], "optimization_pool", 1)[0]
    assert adapter.public_task(row) == public_task(row)
    assert "patch" not in adapter.public_task(row) and "FAIL_TO_PASS" not in adapter.public_task(row)
    closed = []

    @contextlib.contextmanager
    def open_task(actual, *args):
        assert actual is row
        try:
            yield NS(
                task=public_task(row),
                sandbox=NS(sandbox_id="fixture-only"),
                execute=lambda *a, **k: {
                    "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/x b/x",
                    "returncode": 0,
                },
            )
        finally:
            closed.append(True)

    original = runtime.run_agent

    def observed(client, box, visible, *args):
        assert visible == public_task(row)
        return original(client, box, visible, *args)

    def grade(actual, patch, *args):
        assert closed and actual is row and patch == "diff --git a/x b/x"
        if resolved == "infra_failure":
            raise RuntimeError("fixture grader failure")
        return {"score": float(resolved), "resolved": resolved, "test_fixture": True}

    monkeypatch.setattr(sandbox, "open_task", open_task)
    monkeypatch.setattr(runtime, "run_agent", observed)
    monkeypatch.setattr(reward, "grade", grade)
    client = NS(chat=NS(completions=NS(create=lambda **k: completion("submit"))))
    if resolved == "infra_failure":
        with pytest.raises(RuntimeError, match="fixture grader"):
            execution.rollout(client, row, "fixture-policy", c["harness_path"], c, tmp_path, 42)
        assert read(tmp_path / "error.json")["score"] is None
        assert read(tmp_path / "error.json")["sandbox_id"] == "fixture-only"
        assert not (tmp_path / "result.json").exists()
    else:
        result = execution.rollout(client, row, "fixture-policy", c["harness_path"], c, tmp_path, 42)
        assert result["score"] == float(resolved) and result["training_reward"] == (1 if resolved else -1)
        assert result["patch"] == "diff --git a/x b/x" and result["benchmark"] == "swe"
        assert result["sampling_seed"] == 42 and result["harness_revision"] == revision(c["harness_path"])
        assert read(tmp_path / "result.json") == result


def test_unsupported_h_p_fail_before_swe_processing(fixture_config, tmp_path):
    from baseline.audit import preflight
    from baseline.search import h_step
    from baseline.training import prepare

    with pytest.raises(NotImplementedError, match="SWE-specific"):
        h_step(None, None, None, [], [], fixture_config, tmp_path)
    with pytest.raises(NotImplementedError, match="SWE-specific"):
        prepare(None, None, [], fixture_config, tmp_path)
    with pytest.raises(NotImplementedError, match="SWE-specific"):
        preflight(fixture_config)
    with pytest.raises(ValueError, match="Unknown benchmark"):
        benchmarks.get_benchmark({"benchmark": "does-not-exist"})


def test_adapter_source_participates_in_resume_hash(monkeypatch, fixture_config, tmp_path):
    source = tmp_path / "source"
    adapter_file = source / "baseline/benchmarks/fixture.py"
    adapter_file.parent.mkdir(parents=True)
    adapter_file.write_text("VERSION = 1\n")
    prompt = source / "scripts/h_step/propose_prompt.tmpl"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("fixture prompt")
    monkeypatch.setattr(outer_loop, "ROOT", source)
    outer_loop.initialize(fixture_config, tmp_path / "run", test_only=True)
    adapter_file.write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="identical config/code/data"):
        outer_loop.check_state(fixture_config, tmp_path / "run", test_only=True)
