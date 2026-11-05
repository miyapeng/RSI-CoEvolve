"""Offline contract fixtures. Scores here are simulated, not benchmark measurements."""

import asyncio
import enum
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import yaml

from baseline import execution
from baseline.benchmarks import arc_agi_3 as arc_adapter
from baseline.benchmarks import get_benchmark, require_swe
from baseline.benchmarks import osworld as os_adapter
from baseline.benchmarks import vision2web as vision_adapter
from baseline.benchmarks.official import path_value
from baseline.records import ROOT, read, write


def test_official_dispatch_does_not_require_swe_environment(tmp_path):
    # The dedicated Vision2Web Conda environment intentionally has no mini-SWE.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['minisweagent'] = None; from baseline.execution import evaluate",
        ],
        cwd=ROOT,
        check=True,
    )
    # Run the actual entry from an unrelated cwd with Python's path isolation.
    # An unknown fixture adapter stops before native imports or any paid calls.
    request = tmp_path / "request.json"
    write(request, {"source": {"repo": str(tmp_path)}, "benchmark": "bootstrap_fixture"})
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "baseline/benchmarks/official_worker.py"), str(request)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Unknown benchmark adapter: bootstrap_fixture" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def config(name):
    return yaml.safe_load((ROOT / f"configs/benchmarks/{name}.yaml").read_text())


def official_repo(c):
    path = path_value(c["official"]["repo"])
    if not (path / ".git").exists():
        pytest.skip("Optional pinned official source checkout is not present")
    get_benchmark(c).source(c)
    return path


@pytest.fixture
def vision_task(tmp_path):
    c = config("vision2web")
    root = tmp_path / "data"
    project = root / "frontend/example"
    (project / "prototypes").mkdir(parents=True)
    (project / "prototypes/desktop.jpg").write_bytes(b"CPU fixture; no image inference")
    (project / "prompt.txt").write_text("Build a blue button.")
    write(project / "workflow.json", [{"content": [{"validations": ["PRIVATE_WORKFLOW_SENTINEL"]}]}])
    c["official"]["dataset_dir"] = str(root)
    c["model"] = "fixture-model"
    return c


def test_task_projection_pins_and_no_final_test_training(vision_task, tmp_path):
    c = vision_task
    adapter = get_benchmark(c)
    row = adapter.load_tasks(c, "final_test", 1)[0]
    assert row.info.task_id == "frontend/example"
    assert "PRIVATE_WORKFLOW_SENTINEL" not in json.dumps(adapter.public_task(row))
    assert row.public["observations"][1]["artifact"]["media_type"] == "image/jpeg"
    for name in ("vision2web", "osworld_verified", "arc_agi_3"):
        conf = config(name)
        obj = get_benchmark(conf)
        with pytest.raises(ValueError, match="evaluation-only"):
            obj.load_tasks(conf, "optimization_pool", 1)
        with pytest.raises(NotImplementedError, match="SWE-specific"):
            require_swe(conf, "H/P training")
    repo = official_repo(c)
    planned = adapter.plan(row, c["model"], str(repo), c, tmp_path / "run", 42)
    before = planned["task_fingerprint"]
    (Path(c["official"]["dataset_dir"]) / row.info.task_id / "resources").mkdir()
    (Path(c["official"]["dataset_dir"]) / row.info.task_id / "resources/icon.svg").write_text("changed")
    assert adapter.plan(row, c["model"], str(repo), c, tmp_path / "run", 42)["task_fingerprint"] != before
    with pytest.raises(ValueError, match="Policy identity"):
        adapter.plan(row, "other-model", str(repo), c, tmp_path, 42)
    with pytest.raises(ValueError, match="harness path"):
        adapter.plan(row, c["model"], str(tmp_path), c, tmp_path, 42)


def test_official_episode_records_failures_and_no_stale_reuse(monkeypatch, vision_task, tmp_path):
    c = vision_task
    official_repo(c)
    adapter = get_benchmark(c)
    row = adapter.load_tasks(c, "final_test", 1)[0]

    def fixture_worker(command, **kwargs):
        assert "official_worker.py" in command[1]
        request = read(command[2])
        assert request["model"] == c["model"]
        write(
            Path(kwargs["cwd"]) / "native_result.json",
            {
                "task_id": row.info.task_id,
                "score": None,
                "metrics": {"fixture": True, "VS": 0.5, "FS": 1},
                "termination_reason": "fixture",
                "simulated": True,
            },
        )

    # source() uses check_output -> subprocess.run too; leave git calls real.
    real_run = subprocess.run
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kw: real_run(command, **kw) if command[0] == "git" else fixture_worker(command, **kw),
    )
    out = tmp_path / "episode"
    result = execution.rollout(None, row, c["model"], c["official"]["repo"], c, out, 42)
    assert result["simulated"] and result["score"] is None and result["training_reward"] is None
    assert result["sampling_params"]["seed_applied_to_model"] is False
    assert result["benchmark"] == "vision2web"
    with pytest.raises(FileExistsError):
        execution.rollout(None, row, c["model"], c["official"]["repo"], c, out, 42)

    def fail(command, **kw):
        if command[0] == "git":
            return real_run(command, **kw)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        execution.rollout(None, row, c["model"], c["official"]["repo"], c, tmp_path / "failed", 42)
    assert read(tmp_path / "failed/failure.json")["score"] is None
    assert not (tmp_path / "failed/result.json").exists()


def test_vision_native_copy_loop_and_analyzer(monkeypatch, vision_task, tmp_path):
    """Run real official inference orchestration/analysis; replace Docker and API boundaries."""
    c = vision_task
    repo = official_repo(c)
    monkeypatch.syspath_prepend(str(repo))
    from vision2web.evaluation import engine as evaluation
    from vision2web.inference import engine as inference

    closed = []

    class SandboxFixture:
        def __init__(self, **kwargs):
            pass

        async def create_container(self, *a, **kw):
            return "fixture"

        async def start_container(self, *a):
            pass

        async def stop_container(self, *a):
            closed.append(True)

        def get_container_id(self, *a):
            return "fixture"

        async def copy_from_container(self, **kw):
            pass

    class AgentFixture:
        async def run_task(self, workspace, prompt, project_info):
            assert "PRIVATE_WORKFLOW_SENTINEL" not in prompt
            assert not (workspace / "workflow.json").exists()
            assert (workspace / "prompt.txt").read_text() == "Build a blue button."
            (workspace / "start.sh").write_text("# CPU fixture, never executed")
            return {"status": "success"}

    async def browser_fixture(self, **kwargs):
        assert "PRIVATE_WORKFLOW_SENTINEL" in json.dumps(kwargs["workflow_data"])
        work = kwargs["workspace"]
        write(work / "test_results/prototypes/desktop_scores.json", [{"score": 0.6}, {"score": 0.8}])
        write(work / "test_results/workflow_0/test_case_0/result.json", {"result": "Pass"})

    monkeypatch.setattr(inference, "SandboxManager", SandboxFixture)
    monkeypatch.setattr(evaluation, "SandboxManager", SandboxFixture)
    monkeypatch.setattr(inference, "get_adapter", lambda **kwargs: AgentFixture())
    monkeypatch.setattr(evaluation.EvaluationEngine, "_run_evaluation", browser_fixture)
    for key in ("policy_api_key_env", "functional_api_key_env", "visual_api_key_env"):
        monkeypatch.setenv(c["official"][key], "cpu-fixture-unused")
    adapter = get_benchmark(c)
    row = adapter.load_tasks(c, "final_test", 1)[0]
    request = adapter.plan(row, c["model"], str(repo), c, tmp_path / "run", 42)
    result = asyncio.run(vision_adapter.run_native(request))
    assert result["score"] is None  # Do not invent a VS/FS weighted scalar.
    metrics = next(iter(result["metrics"]["analysis"]["results"].values()))
    assert metrics["prototypes"]["desktop"] == pytest.approx(0.7)
    assert metrics["function"]["workflow_0:test_case_0"] == 1
    assert len(closed) == 2  # Both official container lifetimes reached cleanup.


def test_osworld_official_driver_never_passes_evaluator_to_agent(monkeypatch, tmp_path):
    c = config("osworld_verified")
    repo = official_repo(c)
    adapter = get_benchmark(c)
    rows = adapter.load_tasks(c, "final_test", 1000)
    assert len(rows) == 368
    assert all(set(r.public) == {"task_id", "observations"} for r in rows)
    c["official"]["suite"] = "test_nogdrive.json"
    assert len(adapter.load_tasks(c, "final_test", 1000)) == 360
    c["official"]["suite"] = "test_all.json"
    monkeypatch.syspath_prepend(str(repo))
    # Import the real driver without requiring native GUI SDKs; its timeout decorator is unused here.
    monkeypatch.setitem(sys.modules, "wrapt_timeout_decorator", ModuleType("wrapt_timeout_decorator"))
    # Additional, unused OS-Symphony drivers import image helpers at module import time.
    monkeypatch.setitem(sys.modules, "mm_agents.os_symphony.utils.common_utils", NS(draw_coordinates=None))
    monkeypatch.setitem(sys.modules, "mm_agents.os_symphony.utils.process_context", NS(set_current_result_dir=None))
    spec = importlib.util.spec_from_file_location("lib_run_single", repo / "lib_run_single.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setitem(sys.modules, "lib_run_single", driver)
    monkeypatch.setattr(driver.time, "sleep", lambda _: None)
    row = rows[0]
    example = read(row.native["example_path"])
    visible = []
    closed = []

    class Agent:
        action_space = "pyautogui"

        def __init__(self, **kwargs):
            assert kwargs["model"] == c["model"]

        def reset(self, *a, **kw):
            pass

        def predict(self, instruction, obs):
            assert instruction == example["instruction"]
            assert set(obs) == {"screenshot"}
            visible.append(instruction)
            return "fixture done", ["DONE"]

    class Env:
        vm_ip = "fixture"
        controller = NS(start_recording=lambda: None, end_recording=lambda _: None)

        def __init__(self, **kwargs):
            pass

        def reset(self, task_config):
            assert task_config["evaluator"] == example["evaluator"]

        def _get_obs(self):
            return {"screenshot": b"fixture png"}

        def step(self, *a):
            return self._get_obs(), 0, True, {}

        def evaluate(self):
            return 0.25  # CPU verifier fixture, not an OSWorld result.

        def close(self):
            closed.append(True)

    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", NS(DesktopEnv=Env))
    monkeypatch.setitem(sys.modules, "mm_agents.agent", NS(PromptAgent=Agent))
    vm = tmp_path / "fixture.qcow2"
    vm.touch()
    c["official"]["path_to_vm"] = str(vm)
    request = adapter.plan(row, c["model"], str(repo), c, tmp_path / "run", 42)
    monkeypatch.chdir(tmp_path)  # Restored after native wrapper's official relative-path context.
    result = os_adapter.run_native(request)
    assert result["score"] == 0.25 and visible and closed == [True]
    summary = read(tmp_path / "run/native/summary/results.json")
    assert summary[0]["application"] == "chrome"


def test_arc_profile_binding_scorecard_and_thread_failure(monkeypatch, tmp_path):
    c = config("arc_agi_3")
    c["official"]["task_ids"] = ["fixture-123"]
    repo = official_repo(c)
    adapter = get_benchmark(c)
    row = adapter.load_tasks(c, "final_test", 1)[0]
    assert row.public["observations"] == []  # No private game source / solution.
    entry = {"request": {"model": c["model"]}, "client": {}, "runtime": {}}

    class Reason(enum.Enum):
        GAME_WIN = "GAME_WIN"
        ACTION_BUDGET = "ACTION_BUDGET"
        TIME_BUDGET = "TIME_BUDGET"
        AGENT_ERROR = "AGENT_ERROR"

    agent = NS(exit_reason=Reason.GAME_WIN, levels_completed=1, token_counter=10, action_counter=2)
    closed = []

    class Swarm:
        agents = [agent]

        def __init__(self, root, games, **kwargs):
            assert games == ["fixture-123"] and kwargs["config"] == c["official"]["model_config_id"]

        def main(self):
            return NS(model_dump=lambda **kwargs: {"score": 7.5, "card_id": "fixture", "environments": []})

        def request_shutdown(self):
            pass

        def close_scorecard(self):
            closed.append(True)

        def cleanup(self):
            pass

    monkeypatch.setitem(sys.modules, "benchmarking.base", NS(ExitReason=Reason))
    monkeypatch.setitem(sys.modules, "benchmarking.model_config", NS(get_model_config=lambda _: entry))
    monkeypatch.setitem(sys.modules, "benchmarking.runtime_clients", NS(validate_model_credentials=lambda **kw: None))
    monkeypatch.setitem(sys.modules, "benchmarking.swarm", NS(Swarm=Swarm))
    monkeypatch.setenv("ARC_API_KEY", "unused-cpu-fixture")
    monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("ARC_BASE_URL", "https://arcprize.org")
    monkeypatch.delenv("MAX_ACTIONS_HARD_CAP", raising=False)
    request = adapter.plan(row, c["model"], str(repo), c, tmp_path / "run", 42)
    Path(request["output"]).mkdir()
    result = arc_adapter.run_native(request)
    assert result["score"] == 7.5 and result["tokens"] == 10 and closed == [True]
    request["model"] = "wrong-profile"
    with pytest.raises(ValueError, match="profile"):
        arc_adapter.run_native(request)
    request["model"] = c["model"]
    request["output"] = str(tmp_path / "error")
    Path(request["output"]).mkdir()
    agent.exit_reason = Reason.AGENT_ERROR
    with pytest.raises(RuntimeError, match="did not finish normally"):
        arc_adapter.run_native(request)
    assert read(tmp_path / "error/native/scorecard.json")["card_id"] == "fixture"
