"""CPU contracts using pinned upstream sources; no remote/API calls."""

import ast
import asyncio
import importlib.util
import json
import shutil
import subprocess
from io import BytesIO
from types import SimpleNamespace as NS

import pytest
import yaml

from baseline.benchmarks.osworld import OSWorldVerified
from baseline.benchmarks.osworld_tencent import PATCH, RECORDING_PATCH, STREAM_PATCH, provider_pair, source
from baseline.records import ROOT, read


@pytest.fixture
def config():
    config = yaml.safe_load((ROOT / "configs/benchmarks/osworld_tencent.yaml").read_text())
    try:
        OSWorldVerified().source(config)
        source(config["official"])
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("Optional pinned OSWorld / AGS checkouts missing")
    return config


def test_patch_keeps_native_reset_actions_and_verifier(config, tmp_path):
    repo = OSWorldVerified().source(config)["repo"]
    original = ROOT / repo / "desktop_env/desktop_env.py"
    target = tmp_path / "desktop_env.py"
    shutil.copyfile(original, target)
    subprocess.run(["patch", "--batch", "--fuzz=0", str(target), str(PATCH)], check=True, capture_output=True)

    def methods(path):
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}

    before, after = methods(original), methods(target)
    assert before.pop("__init__") != after.pop("__init__")
    assert before == after
    assert {"reset", "step", "evaluate", "_start_emulator"} <= before.keys()
    adapter = OSWorldVerified()
    task = adapter.load_tasks(config, "final_test", 1)[0]
    plan = adapter.plan(task, config["model"], repo, config, tmp_path / "episode", 42)
    assert plan["options"]["tencent_source"]["provider_patch_sha256"]
    assert "evaluator" not in json.dumps(adapter.public_task(task))


def test_provider_recreates_on_reset_and_reports_failed_cleanup(config, monkeypatch, tmp_path):
    import e2b

    monkeypatch.syspath_prepend(OSWorldVerified().source(config)["repo"])
    import desktop_env.providers

    ags_path = str(
        ROOT / source(config["official"])["repo"] / "examples/osworld-ags/overlay/OSWorld/desktop_env/providers"
    )
    monkeypatch.setattr(desktop_env.providers, "__path__", [*desktop_env.providers.__path__, ags_path])
    from desktop_env.providers.ags.provider import AGSProvider

    monkeypatch.setenv("E2B_API_KEY", "unused-cpu-fixture")
    calls, instances = [], []

    class SandboxFixture:
        _envd_access_token = "unused-fixture-token"
        fail_kill = False

        def __init__(self):
            self.sandbox_id = f"fixture-{len(instances)}"
            self.killed = False

        def get_info(self):
            return NS(template_id="sdt-m63gjp45", cpu_count=4, memory_mb=8192)

        def kill(self):
            if self.fail_kill:
                raise RuntimeError("fixture cleanup failure")
            self.killed = True

    def create(**kw):
        calls.append(kw)
        sbx = SandboxFixture()
        instances.append(sbx)
        return sbx

    monkeypatch.setattr(e2b.Sandbox, "create", create)
    for method in ("_start_local_proxies", "_stop_local_proxies", "_wait_for_vm_ready", "_ensure_chromium_cdp_bridge"):
        monkeypatch.setattr(AGSProvider, method, lambda *a, **kw: None)
    _, provider = provider_pair(config["official"], tmp_path)
    provider.start_emulator("fixture", True)
    with pytest.raises(RuntimeError, match="reuse"):
        provider.start_emulator("fixture", True)
    provider.revert_to_snapshot("fixture", "clean")
    assert instances[0].killed
    provider.start_emulator("fixture", True)
    assert len(instances) == 2 and provider.sandbox_id != instances[0].sandbox_id
    assert calls[0]["secure"] and calls[0]["domain"] == "ap-guangzhou.tencentags.com"
    instances[1].fail_kill = True
    with pytest.raises(RuntimeError, match="cleanup failure"):
        provider.stop_emulator(None)
    assert provider.sandbox is instances[1]  # retain identity for recovery
    instances[1].fail_kill = False
    provider.stop_emulator(None)
    provider.stop_emulator(None)
    events = [json.loads(line) for line in (tmp_path / "sandbox_lifecycle.jsonl").read_text().splitlines()]
    assert [e["success"] for e in events if e["operation"] == "cleanup"] == [True, False, True]


def test_gateway_uses_native_prompt_parser_and_image_artifacts(config, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(OSWorldVerified().source(config)["repo"])
    # Native model/grader packages live in the dedicated rsi-osworld Conda env.
    pytest.importorskip("dashscope")
    from mm_agents.agent import PromptAgent
    from openai import OpenAI
    from PIL import Image

    from baseline.benchmarks.osworld_gateway import make_agent

    monkeypatch.setenv("LLM_CENTER_API_KEY", "unused-cpu-fixture")
    agent, client = make_agent(config["official"], "fixture-model", tmp_path)
    calls = []

    def create(**payload):
        calls.append(payload)
        return NS(
            choices=[NS(message=NS(content="```python\npyautogui.click(10, 20)\n```"))],
            model="fixture-returned-model",
            usage=None,
            model_dump=lambda **kw: {"model": "fixture-returned-model", "simulated": True},
        )

    monkeypatch.setattr(client.chat.completions, "create", create)
    buf = BytesIO()
    Image.new("RGB", (20, 20)).save(buf, format="PNG")
    agent.reset()
    _, actions = agent.predict(
        "Click the public button.", {"screenshot": buf.getvalue(), "evaluator": "PRIVATE_SENTINEL"}
    )
    assert actions == ["pyautogui.click(10, 20)"]
    assert type(agent).predict is PromptAgent.predict
    assert "PRIVATE_SENTINEL" not in json.dumps(calls)
    assert "data:image/png;base64," in json.dumps(calls)
    record = read(tmp_path / "model_calls/request_001.json")
    assert "data:image" not in json.dumps(record)
    assert len(list((tmp_path / "model_calls").glob("*.png"))) == 1
    assert agent.returned_models == ["fixture-returned-model"] and agent.usage == [None]
    assert isinstance(client, OpenAI)
    client.close()


def test_proxy_streams_original_bytes_without_buffering(config, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(OSWorldVerified().source(config)["repo"])
    import desktop_env.providers

    original = ROOT / source(config["official"])["repo"] / "examples/osworld-ags/overlay/OSWorld/desktop_env/providers"
    monkeypatch.setattr(desktop_env.providers, "__path__", [*desktop_env.providers.__path__, str(original)])
    target = tmp_path / "provider.py"
    shutil.copyfile(original / "ags/provider.py", target)
    subprocess.run(["patch", "--batch", "--fuzz=0", str(target), str(STREAM_PATCH)], check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location("streaming_proxy_fixture", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []

    class Context:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self.value

        async def __aexit__(self, *args):
            pass

    class Content:
        async def iter_chunked(self, size):
            for chunk in (b"original", b"-png-bytes"):
                events.append(("read", chunk))
                yield chunk

    class Stream:
        def __init__(self, **kw):
            assert kw["status"] == 200 and kw["headers"] == {"Content-Type": "image/png"}

        async def prepare(self, request):
            events.append(("prepare", None))

        async def write(self, chunk):
            events.append(("write", chunk))

        async def write_eof(self):
            events.append(("eof", None))

    remote = NS(status=200, headers={"Content-Type": "image/png", "Content-Length": "18"}, content=Content())
    session = NS(request=lambda *a, **kw: Context(remote))
    monkeypatch.setattr(module.aiohttp, "ClientSession", lambda: Context(session))
    monkeypatch.setattr(module.web, "StreamResponse", Stream)

    async def empty_body():
        return b""

    request = NS(method="GET", headers={}, read=empty_body)
    proxy = module.LocalProxyServer(15000, "unused-fixture", "unused-fixture-token")
    asyncio.run(proxy._handle_http(request, "/screenshot"))
    assert events == [
        ("prepare", None),
        ("read", b"original"),
        ("write", b"original"),
        ("read", b"-png-bytes"),
        ("write", b"-png-bytes"),
        ("eof", None),
    ]


def test_recording_launcher_patch_keeps_official_transaction_and_routes(config, tmp_path):
    repo = ROOT / OSWorldVerified().source(config)["repo"] / "desktop_env"
    target = tmp_path / "guest.py"
    shutil.copyfile(repo / "recording_update/guest.py", target)
    subprocess.run(["patch", "--batch", "--fuzz=0", str(target), str(RECORDING_PATCH)], check=True, capture_output=True)
    scope = {"__name__": "fixture_guest"}
    exec(compile(target.read_text(), str(target), "exec"), scope)
    launcher = 'from src.server import app, asgi_app, main\n\n\nif __name__ == "__main__":\n    main()\n'
    scope["legacy_region"](launcher)
    with pytest.raises(RuntimeError, match="Unrecognized"):
        scope["legacy_region"](launcher + "# different template version\n")
    recorder = {}
    exec(compile((repo / "server/screen_recording.py").read_text(), "recorder.py", "exec"), recorder)
    routes = recorder["recording_routes"]((repo / "server/main.py").read_text())
    result = scope["patch_source"](launcher, routes)
    assert recorder["recording_routes"](result) == routes
    assert "src/http_routes.py" in result  # inspected server hashes checked before startup
    before = ast.parse((repo / "recording_update/guest.py").read_text())
    after = ast.parse(target.read_text())
    for name in ("discover", "apply_update", "healthy", "atomic_write"):
        assert ast.dump(next(n for n in before.body if getattr(n, "name", None) == name)) == ast.dump(
            next(n for n in after.body if getattr(n, "name", None) == name)
        )
