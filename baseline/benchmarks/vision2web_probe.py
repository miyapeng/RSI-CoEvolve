"""One real Tencent transport/browser probe, without model calls or benchmark scores."""

import argparse
import asyncio
import getpass
import json
import os
import shlex
import sys
import time
from pathlib import Path

import yaml

from baseline.records import digest, sha, write

from .official import path_value
from .vision2web import Vision2Web
from .vision2web_e2b import transport, validate


async def probe(request, transport_only):
    from vision2web.core.sandbox import SandboxManager

    out = Path(request["output"])
    workspace = out / "workspace"
    workspace.mkdir()
    payload = bytes(range(256)) + b"\x00binary-roundtrip\xff"
    (workspace / "binary.dat").write_bytes(payload)
    manager = SandboxManager(image_name=request["options"]["sandbox_image"])
    checks = {}
    try:
        iid = await manager.create_container(workspace)
        if not iid or not await manager.start_container(workspace):
            raise RuntimeError("Official SandboxManager could not create/start Tencent sandbox")
        checks["sandbox_id"] = iid
        code, stdout, _ = await manager.exec_command(workspace, "printf VISION2WEB_TRANSPORT_OK")
        assert code == 0 and stdout == "VISION2WEB_TRANSPORT_OK"
        checks["official_exec"] = True
        # Test the official Claude adapter's stdin protocol, without invoking a model.
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            iid,
            "sh",
            "-c",
            "cat > /workspace/stdin.dat",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate(payload)
        assert process.returncode == 0
        # Same detached website start and localhost routing used by the native evaluator.
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-d",
            "-w",
            "/workspace",
            iid,
            "sh",
            "-c",
            "python3 -m http.server 3000 > /tmp/probe-server.log 2>&1",
        )
        assert await process.wait() == 0
        # A detached process may return before its listening socket is ready.
        readiness = """import time, urllib.request, urllib.error
deadline = time.monotonic() + 10
while True:
    try:
        response = urllib.request.urlopen('http://localhost:3000/binary.dat', timeout=2).read()
        break
    except urllib.error.URLError:
        if time.monotonic() >= deadline:
            raise
        time.sleep(0.1)
assert response == open('binary.dat', 'rb').read()
"""
        code, _, _ = await manager.exec_command(workspace, "python3 -c " + shlex.quote(readiness))
        assert code == 0
        checks["detached_website"] = True
        browser = "NOT_RUN"
        if not transport_only:
            code, stdout, _ = await manager.exec_command(
                workspace, "node --version && claude --version && playwright-cli --version && google-chrome --version"
            )
            checks["tool_versions"] = stdout
            if code:
                raise RuntimeError("Image lacks official Vision2Web runtime tools")
            code, _, _ = await manager.exec_command(
                workspace,
                "playwright-cli open http://localhost:3000 && playwright-cli screenshot --filename=/workspace/browser.png && playwright-cli close",
            )
            if code:
                raise RuntimeError("Real browser screenshot failed")
            browser = "PASS"
        downloaded = out / "downloaded"
        assert await manager.copy_from_container(workspace, host_path=downloaded)
        assert (downloaded / "binary.dat").read_bytes() == payload
        assert (downloaded / "stdin.dat").read_bytes() == payload
        if not transport_only:
            assert (downloaded / "browser.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        checks["binary_upload_stdin_download"] = True
        return {"checks": checks, "browser": browser}
    finally:
        await manager.cleanup_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/vision2web_tencent.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--transport-only", action="store_true", help="No browser/tools check; never counts as benchmark readiness"
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    options = config["official"]
    options["sandbox_image"] = os.path.expandvars(options["sandbox_image"])
    cfg = validate(options)
    cfg["sandbox_timeout"] = 600
    cfg["command_timeout"] = 90
    if args.transport_only:
        cfg.update(cpu=1, memory="1Gi")
    source = Vision2Web().source(config)
    if not args.execute:
        print(
            json.dumps(
                {
                    "image": options["sandbox_image"],
                    "tencent": cfg,
                    "execute": False,
                    "transport_only": args.transport_only,
                },
                indent=2,
            )
        )
        return 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not os.environ.get(cfg["key_env"]):
        os.environ[cfg["key_env"]] = getpass.getpass(f"{cfg['key_env']} (hidden): ")
    sys.path.insert(0, str(path_value(options["repo"])))
    report = {
        "simulated": False,
        "source": source,
        "adapter_code_hash": digest({p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}),
        "image": options["sandbox_image"],
        "model_calls": 0,
        "benchmark": "NOT_RUN",
        "official_scoring": "NOT_RUN",
        "browser": "NOT_RUN",
    }
    started = time.monotonic()
    try:
        with transport({"options": options, "output": str(out)}):
            report.update(asyncio.run(probe({"options": options, "output": str(out)}, args.transport_only)))
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error_type=type(exc).__name__)
    finally:
        states = [json.loads(p.read_text()) for p in (out / "transport").glob("sandbox-*.json")]
        report["cleanup"] = bool(states) and all(s["stopped"] for s in states)
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write(out / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
