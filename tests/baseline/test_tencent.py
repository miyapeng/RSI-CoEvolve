"""Tencent protocol tests with scripted SDK responses; no live sandbox or benchmark scores."""

import importlib
from functools import partial
from types import SimpleNamespace as NS

import pytest

from baseline import tencent
from baseline.config import load_config
from baseline.records import ROOT
from baseline.sandbox import GraderEnvironment, TaskSandbox


def task():
    return {
        "task_id": "fixture-task",
        "image": "source/image:task",
        "base_commit": "a" * 40,
        "cwd": "/testbed",
        "split": "optimization_pool",
        "dataset_source": "fixture",
        "problem_statement": "fixture",
    }


@pytest.fixture
def sdk(monkeypatch):
    import e2b
    from e2b.sandbox.commands.command_handle import CommandExitException

    cfg = load_config(ROOT / "configs/baseline_9b.yaml")["runtime"]["sandbox"]
    cfg["images"] = {task()["image"]: {"image": "registry.example/swe/task:fixed", "digest": "sha256:" + "a" * 64}}
    for key in tencent.required_env(cfg):
        monkeypatch.setenv(key, "fixture-value")
    monkeypatch.setattr(tencent.shutil, "which", lambda _: "/fixture/skopeo")
    monkeypatch.setattr(tencent, "check_image", lambda _: None)
    calls, commands, uploads, killed = [], [], [], []

    def command(command, **kw):
        commands.append((command, kw))
        if "transport-fail" in command:
            raise ConnectionError("fixture transport failure")
        if "false-fixture" in command:
            raise CommandExitException(stdout="", stderr="failure", exit_code=2, error=None)
        output = ""
        if "id -u" in command:
            output = "65534\n"
        if "eval.sh" in command or "stdout.log" in command:
            output = ">>>>> Start Test Output\nPASSED test_fixture\n>>>>> End Test Output\n"
        if "__APPLY_OK__" in command:
            output = "__APPLY_OK__"
        if "reward.txt" in command:
            output = "1"
        return NS(stdout=output, stderr="", exit_code=0)

    box = NS(
        sandbox_id="fixture-id",
        commands=NS(run=command),
        files=NS(write=lambda *a, **k: uploads.append((a, k))),
        kill=lambda: killed.append(True),
    )
    monkeypatch.setattr(e2b.Sandbox, "create", lambda **kw: (calls.append(("create", kw)), box)[1])
    monkeypatch.setattr(e2b.Sandbox, "connect", lambda *a, **kw: (calls.append(("connect", kw)), box)[1])
    return NS(config=cfg, calls=calls, commands=commands, uploads=uploads, killed=killed)


@pytest.mark.parametrize("connection", ["proxy", "direct"])
def test_tencent_task_transport_isolation_exit_status_and_cleanup(sdk, connection):
    cfg = {**sdk.config, "connection": connection}
    with TaskSandbox(task(), cfg) as box:
        assert box.execute({"command": "false-fixture"})["returncode"] == 2
        with pytest.raises(ConnectionError):
            box.raw("transport-fail")
        box.execute({"command": "echo task"}, timeout=7)
        assert box.execution_image == "registry.example/swe/task:fixed"
    create = sdk.calls[0][1]
    assert create["allow_internet_access"] is False and create["secure"]
    assert create["metadata"]["task_id"] == "fixture-task" and "job_id" in create["metadata"]
    assert "envs" not in create and "volume_mounts" not in create
    if connection == "direct":
        assert "proxy" not in create
        assert sdk.calls[1][1]["sandbox_url"] == "https://49983-fixture-id.fixture-value"
    else:
        assert create["proxy"] == "fixture-value" and len(sdk.calls) == 1
    action = sdk.commands[-1][0]
    assert "setpriv" in action and "--no-new-privs" in action and "env -i" in action
    assert "timeout -k 5 7" in action and "fixture-value" not in str(sdk.commands)
    assert sdk.killed == [True]


def test_failed_reconnect_cleanup_and_pinned_image_check(sdk, monkeypatch):
    import e2b

    partial_manifest = {**sdk.config, "images": {**sdk.config["images"], "unused/image:tag": None}}
    assert not tencent.preflight(partial_manifest, [task()["image"]])["blockers"]
    assert tencent.preflight(partial_manifest, ["missing/image:tag"])["missing_images"] == ["missing/image:tag"]

    def fail(*a, **k):
        raise ConnectionError("reconnect failed")

    monkeypatch.setattr(e2b.Sandbox, "connect", fail)
    with pytest.raises(ConnectionError), TaskSandbox(task(), {**sdk.config, "connection": "direct"}):
        pass
    assert sdk.killed == [True]
    # Restore only the real checker, leaving the SDK fixture disconnected.
    monkeypatch.undo()
    monkeypatch.setattr(tencent.subprocess, "run", lambda *a, **k: NS(returncode=0, stdout="sha256:drift"))
    with pytest.raises(ValueError, match="drifted"):
        tencent.check_image(sdk.config["images"][task()["image"]])
    with pytest.raises(ValueError, match="environment-variable"):
        tencent.validate_config({**sdk.config, "api_key": "do-not-store"})


@pytest.mark.parametrize(
    "name", ["swebench_verify_orchard", "swebench_multilingual_verify_orchard", "swebench_pro_verify_orchard"]
)
def test_original_final_grader_runs_through_explicit_tencent_environment(name, sdk, monkeypatch, tmp_path):
    module = importlib.import_module("minisweagent.run.benchmarks." + name)
    row = {
        "instance_id": task()["task_id"],
        "image_name": task()["image"],
        "repo": "fixture/repo",
        "FAIL_TO_PASS": ["test_fixture"],
        "PASS_TO_PASS": [],
        "test_patch": "fixture test patch",
        "patch": "SECRET_GOLD_MUST_NOT_BE_UPLOADED",
    }
    if name == "swebench_pro_verify_orchard":
        folder = tmp_path / "official-tests"
        (folder / "tests").mkdir(parents=True)
        (folder / "tests/test.sh").write_text("echo official fixture verifier")
        monkeypatch.setattr(module, "_task_dir", lambda _: folder)
        monkeypatch.setenv("SWE_PRO_ENTRY_TIMEOUT", "30")
    else:
        monkeypatch.setattr(module, "get_swebench_docker_image_name", lambda _: task()["image"])
        if name == "swebench_verify_orchard":
            monkeypatch.setattr(module, "make_test_spec", lambda _: NS(eval_script="echo fixture-tests"))
            monkeypatch.setattr(module, "get_eval_report", lambda **k: {task()["task_id"]: {"resolved": True}})
        else:
            monkeypatch.setattr(module, "_build_eval_script", lambda _: "echo fixture-tests")
            monkeypatch.setattr(module, "_parse_test_output", lambda *a: {"test_fixture": "PASSED"})
    result = module.evaluate_instance_orchard(
        row,
        "fixture predicted patch",
        {},
        logs_dir=tmp_path / "logs",
        environment_factory=partial(GraderEnvironment, task(), sdk.config),
    )
    assert result["resolved"] and not result["error"]
    assert sdk.calls[0][1]["allow_internet_access"] is True and sdk.killed == [True]
    assert "SECRET_GOLD_MUST_NOT_BE_UPLOADED" not in str(sdk.commands) + str(sdk.uploads)
