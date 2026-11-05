"""Transport contract tests only; fake SDK output is not benchmark performance."""

import io
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from baseline.benchmarks import vision2web_e2b as bridge
from baseline.benchmarks.vision2web import evaluation_artifacts
from baseline.records import ROOT, read, write


def options():
    c = yaml.safe_load((ROOT / "configs/benchmarks/vision2web_tencent.yaml").read_text())["official"]
    c["sandbox_image"] = "modelbest.tencentcloudcr.com/testing/web:v1@sha256:" + "a" * 64
    return c


def tar_bytes(entries):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        for entry, data in entries:
            archive.addfile(entry, io.BytesIO(data) if data else None)
    return buf.getvalue()


def test_official_command_surface_and_secret_boundary(monkeypatch, tmp_path):
    import e2b

    calls, created, killed, files = [], [], [], {}
    member = tarfile.TarInfo("site/screenshot.png")
    pixels = b"\x89PNG\r\n\x00\xff"
    member.size = len(pixels)
    archive = tar_bytes([(member, pixels)])

    def command(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return NS(exit_code=7 if cmd == "false" else 0, stdout="observed", stderr="", pid=123)

    box = NS(
        sandbox_id="fixture123",
        commands=NS(run=command),
        files=NS(write=lambda p, data, **kw: files.update({p: data}), read=lambda *a, **kw: archive),
        kill=lambda: killed.append(True),
    )
    monkeypatch.setattr(e2b.Sandbox, "create", lambda **kw: created.append(kw) or box)
    monkeypatch.setattr(e2b.Sandbox, "connect", lambda *a, **kw: box)
    # Dispatch contract only; the real foreground file protocol is exercised below.
    monkeypatch.setattr(
        bridge,
        "run_foreground",
        lambda b, c, cmd, env, cwd, user: b.commands.run(cmd, envs=env, cwd=cwd, user=user),
    )
    monkeypatch.setenv("E2B_API_KEY", "HOST_SANDBOX_KEY_SENTINEL")
    monkeypatch.setenv("http_proxy", "http://host-only-proxy")
    c = options()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "site.txt").write_text("public")
    root = tmp_path / "transport"

    def invoke(*args, data=b""):
        out, err = io.BytesIO(), io.BytesIO()
        code = bridge.dispatch(root, list(args), io.BytesIO(data), out, err)
        return code, out.getvalue(), err.getvalue()

    with bridge.transport({"options": c, "output": str(tmp_path)}):
        assert "http_proxy" not in os.environ
        assert invoke("images", "-q", c["sandbox_image"])[0] == 0
        assert (
            invoke(
                "create",
                "--name",
                "vision2web-test",
                "--workdir",
                "/workspace",
                "--rm",
                "--memory",
                "64g",
                "--memory-swap",
                "64g",
                "-e",
                "PLAYWRIGHT_MCP_SANDBOX=false",
                c["sandbox_image"],
                "sleep",
                "infinity",
            )[1].strip()
            == b"fixture123"
        )
        assert invoke("start", "fixture123")[0] == 0
        assert invoke("cp", str(workspace) + "/.", "fixture123:/tmp/workspace")[0] == 0
        assert (
            invoke(
                "exec",
                "-i",
                "-w",
                "/workspace",
                "-e",
                "ANTHROPIC_AUTH_TOKEN=POLICY_SECRET",
                "fixture123",
                "sh",
                "-c",
                "cat > /tmp/prompt",
                data=pixels,
            )[0]
            == 0
        )
        assert pixels in files.values()
        assert any(kw["envs"].get("ANTHROPIC_AUTH_TOKEN") == "POLICY_SECRET" for _, kw in calls if "envs" in kw)
        assert "HOST_SANDBOX_KEY_SENTINEL" not in json.dumps(calls)
        assert invoke("exec", "fixture123", "false")[0] == 7
        assert invoke("exec", "-d", "fixture123", "bash", "start.sh")[0] == 0
        assert calls[-1][1]["background"]
        assert calls[-1][1]["request_timeout"] == c["tencent"]["request_timeout"]
        assert invoke("exec", "fixture123", "tar", "-C", "/workspace", "-cf", "-", ".")[1] == archive
        assert invoke("cp", "fixture123:/workspace/site", str(tmp_path / "download"))[0] == 0
        assert (tmp_path / "download/site/screenshot.png").read_bytes() == pixels
        with pytest.raises(ValueError, match="escapes"):
            invoke("cp", "/etc/passwd", "fixture123:/tmp/workspace")
        with pytest.raises(ValueError, match="Unsupported"):
            invoke("run", "unexpected-image")
        assert invoke("stop", "fixture123")[0] == 0
    assert os.environ["http_proxy"] == "http://host-only-proxy"
    assert len(killed) == 1
    custom = json.loads(created[0]["metadata"]["x-custom-config"])
    assert custom["resources"] == {"cpu": "2", "memory": "8Gi"}
    assert created[0]["allow_internet_access"] is True
    assert read(root / "sandbox-fixture123.json")["stopped"]
    assert "POLICY_SECRET" not in (root / "transport.jsonl").read_text()
    assert "HOST_SANDBOX_KEY_SENTINEL" not in (root / "settings.json").read_text()


def test_foreground_preserves_outputs_stdin_exit_and_process_deadline(tmp_path):
    # Exercise the shell/file protocol with real local processes, no remote/API.
    def command(cmd, background=False, **kw):
        args = dict(shell=True, cwd=kw.get("cwd", tmp_path), env={**os.environ, **kw.get("envs", {})})
        if background:
            process = subprocess.Popen(cmd, **args)
            return NS(disconnect=lambda: None, kill=process.kill)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=kw["timeout"], **args)
        return NS(stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode)

    box = NS(
        commands=NS(run=command),
        files=NS(
            write=lambda path, value, **kw: (
                Path(path).write_bytes(value) if isinstance(value, bytes) else Path(path).write_text(value)
            ),
            read=lambda path, **kw: Path(path).read_text(),
        ),
    )
    cfg = options()["tencent"]
    result = bridge.run_command(
        box,
        cfg,
        ["bash", "-c", 'cat; printf "%s" "$FIXTURE" >&2; exit 7'],
        env={"FIXTURE": "exact stderr"},
        cwd=str(tmp_path),
        stdin=b"exact stdin\n",
    )
    assert (result.exit_code, result.stdout, result.stderr) == (7, "exact stdin\n", "exact stderr")
    cfg["command_timeout"] = 0.2
    result = bridge.run_command(box, cfg, ["sleep", "5"], cwd=str(tmp_path))
    assert result.exit_code == 124  # GNU timeout actually ended the process.


def test_cleanup_on_native_failure_and_archive_validation(monkeypatch, tmp_path):
    killed = []
    monkeypatch.setattr(bridge, "connect", lambda *a: NS(kill=lambda: killed.append(True)))
    original_path = os.environ["PATH"]
    with pytest.raises(RuntimeError, match="native failure"):
        with bridge.transport({"options": options(), "output": str(tmp_path)}):
            write(tmp_path / "transport/sandbox-fixture.json", {"sandbox_id": "fixture", "stopped": False})
            raise RuntimeError("native failure")
    assert killed == [True] and os.environ["PATH"] == original_path
    bad = tarfile.TarInfo("../../outside")
    with pytest.raises(tarfile.FilterError):
        bridge.checked_archive(tar_bytes([(bad, b"")]))
    with pytest.raises(ValueError, match="Absolute"):
        bridge.checked_archive(tar_bytes([(tarfile.TarInfo("/absolute"), b"")]))
    privileged = tarfile.TarInfo("executable")
    privileged.mode = 0o4755
    with pytest.raises(ValueError, match="privileged"):
        bridge.checked_archive(tar_bytes([(privileged, b"")]))
    link = tarfile.TarInfo("a")
    link.type, link.linkname = tarfile.SYMTYPE, "."
    nested = tarfile.TarInfo("a/b")
    nested.type, nested.linkname = tarfile.SYMTYPE, "../outside"
    with pytest.raises(ValueError, match="traverses"):
        bridge.checked_archive(tar_bytes([(link, b""), (nested, b"")]))
    with pytest.raises(ValueError, match="Duplicate"):
        bridge.checked_archive(tar_bytes([(tarfile.TarInfo("same"), b""), (tarfile.TarInfo("same"), b"")]))


def test_config_pins_and_missing_verifier_evidence(tmp_path, monkeypatch):
    c = options()
    monkeypatch.setenv("WEB_IMAGE", c["sandbox_image"])
    c["sandbox_image"] = "${WEB_IMAGE}"
    bridge.validate(c)
    assert "${" not in c["sandbox_image"]
    c["sandbox_image"] = "modelbest.tencentcloudcr.com/testing/web:latest"
    with pytest.raises(ValueError, match="pinned"):
        bridge.validate(c)
    project = tmp_path / "project"
    work = tmp_path / "result"
    write(project / "workflow.json", [{"prototype": {"desktop": {}}, "content": [{"validations": ["private"]}]}])
    result = evaluation_artifacts(work, NS(path=project))
    assert not result["complete"] and len(result["missing"]) == 2
    # An official status=success alone is not evidence that either judge completed.
    write(work / "evaluation_result.json", {"status": "success"})
    assert not evaluation_artifacts(work, NS(path=project))["complete"]
    write(work / "test_results/prototypes/desktop_scores.json", [{"score": 0.5}])
    write(work / "test_results/workflow_0/test_case_0/result.json", {"result": "Fail"})
    assert evaluation_artifacts(work, NS(path=project))["complete"]  # A real failure is valid evidence.


def test_proxy_stays_on_host_and_remote_preflight_does_not_require_docker(monkeypatch, tmp_path):
    from baseline.benchmarks import __main__ as cli

    c = options()
    c["tencent"]["connection"] = "proxy"
    monkeypatch.setenv("http_proxy", "http://host-only-proxy")
    with bridge.transport({"options": c, "output": str(tmp_path)}):
        internal = read(tmp_path / "transport/settings.json")["config"]
        assert "http_proxy" not in os.environ
        assert os.environ[internal["proxy_env"]] == "http://host-only-proxy"
    assert os.environ["http_proxy"] == "http://host-only-proxy"
    for name in ("E2B_API_KEY", "ANTHROPIC_API_KEY", "VISION2WEB_VISUAL_API_KEY"):
        monkeypatch.setenv(name, "test-placeholder")
    monkeypatch.setattr(
        cli,
        "get_benchmark",
        lambda config: NS(
            name="vision2web",
            source=lambda c: {},
            load_tasks=lambda *a: [NS(info=NS(task_id="fixture"))],
            evaluation_role=lambda c: "final_test",
        ),
    )
    commands = []

    def run(command, **kw):
        commands.append(command)
        return NS(returncode=0, stdout="{}" if "importlib.import_module" in command[2] else bridge.SDK_VERSION)

    monkeypatch.setattr(cli.subprocess, "run", run)
    report = cli.preflight({"benchmark": "vision2web", "official": c})
    assert report["ready_for_live_attempt"]
    assert "docker_image" not in report["checks"]
    assert len(commands) == 2 and all(cmd[0] == "python" for cmd in commands)
