"""The shared local SGLang server and a sandbox-independent GPU smoke entry."""

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI

from .records import write


def serve_command(checkpoint, config):
    c, r = config["serving"], config["runtime"]
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        str(c["port"]),
        "--tp",
        str(c["tensor_parallel"]),
        "--context-length",
        str(r["max_context_tokens"]),
        "--attention-backend",
        c["attention_backend"],
        "--mem-fraction-static",
        str(c["mem_fraction_static"]),
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
        "--mamba-scheduler-strategy",
        "extra_buffer",
    ]


def tool_smoke(client, checkpoint, output):
    # Official benchmark dispatch can import this module in its own Conda
    # environment; only the SWE serving smoke needs mini-SWE's action schema.
    from minisweagent.models.utils.actions_toolcall import TOOLS

    r = client.chat.completions.create(
        model=checkpoint,
        messages=[{"role": "user", "content": "Call bash with command exactly: printf baseline_tool_smoke"}],
        tools=TOOLS,
        temperature=0,
        max_tokens=512,
        timeout=60,
    )
    write(output, r.model_dump(mode="json"))
    calls = r.choices[0].message.tool_calls or []
    if not calls or any(c.function.name != "bash" for c in calls):
        raise RuntimeError("Serving works but the real tool-call smoke failed")
    import json

    if not all(isinstance(json.loads(c.function.arguments).get("command"), str) for c in calls):
        raise RuntimeError("Malformed tool-call arguments")


@contextlib.contextmanager
def serve(checkpoint, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    port = config["serving"]["port"]
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError("Serving port occupied; refusing to use an unattested existing policy server")
    cmd = serve_command(checkpoint, config)
    write(output / "serve-command.json", {"command": cmd, "checkpoint": checkpoint})
    with (output / "serve.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + config["serving"]["startup_timeout"]
            with httpx.Client(trust_env=False, timeout=2) as http:
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError("Policy server exited; see serve.log")
                    try:
                        if http.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Policy server startup timed out")
                    time.sleep(1)
            with OpenAI(
                base_url=f"http://127.0.0.1:{port}/v1",
                api_key="local",
                max_retries=0,
                http_client=httpx.Client(trust_env=False),
            ) as client:
                tool_smoke(client, checkpoint, output / "tool-smoke.json")
                yield client
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def main():
    """Exercise exactly the server/tool parser used by rollout and post-training eval."""
    import argparse

    import torch

    from .config import load_config
    from .records import sha

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "simulated": False,
        "job_id": os.environ.get("JOB_ID"),
        "config_sha256": sha(args.config),
        "policy_checkpoint": config["model_checkpoint"],
        "grpo_update": "NOT_RUN",
        "task_rollout": "NOT_RUN",
        "status": "FAILED",
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    t0 = time.monotonic()
    try:
        report["image_sources"] = {}
        for root in ("/root/Megatron-LM", "/opt/Megatron-LM", "/workspace/Megatron-LM", "/root/slime"):
            path = Path(root)
            if path.exists():
                head = subprocess.run(
                    ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
                )
                report["image_sources"][root] = {
                    "commit": head.stdout.strip() if head.returncode == 0 else None,
                    "megatron_training_exists": (path / "megatron/training/arguments.py").exists(),
                }
        if not torch.cuda.is_available() or torch.cuda.device_count() < config["serving"]["tensor_parallel"]:
            raise RuntimeError("Insufficient visible GPUs for the configured serving tensor parallel size")
        report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        report["driver"] = driver.stdout.strip().splitlines()
        with serve(config["model_checkpoint"], config, args.output / "server"):
            pass
        report["status"] = "PASSED"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        raise
    finally:
        report["seconds"] = round(time.monotonic() - t0, 3)
        write(args.output / "report.json", report)
        print(report, flush=True)


if __name__ == "__main__":
    main()
