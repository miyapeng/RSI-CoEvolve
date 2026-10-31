"""Own one policy server at a time; execute and score with the same full mini agent."""

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from minisweagent.models.utils.actions_toolcall import TOOLS
from openai import OpenAI

from .data import public_task
from .harness import revision
from .records import append, digest, write
from .reward import grade
from .runtime import run_agent
from .sandbox import open_task


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
    r = client.chat.completions.create(
        model=checkpoint,
        messages=[{"role": "user", "content": "Call bash with command exactly: printf baseline_tool_smoke"}],
        tools=TOOLS,
        temperature=0,
        max_tokens=512,
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


def rollout(client, row, checkpoint, harness, config, output, seed):
    output = Path(output)
    c = {**config["runtime"], "sampling_seed": seed}
    try:
        with open_task(row, c["sandbox"]) as sandbox:
            result = run_agent(client, sandbox, public_task(row), harness, checkpoint, c, output)
        reward = grade(row, result["patch"], c, output / "grading")
        result["score"] = reward["score"]
        # Self-HPO simple reward mode: solved +1, unresolved -1.
        result["training_reward"] = 1.0 if reward["resolved"] else -1.0
        result["reward"] = reward
        write(output / "result.json", result)
        return result
    except Exception as exc:
        write(
            output / "error.json",
            {
                "task_id": row["metadata"]["instance_id"],
                "score": None,
                "policy_checkpoint": checkpoint,
                "harness_revision": revision(harness),
                "error_type": type(exc).__name__,
                "split": row["split"],
            },
        )
        raise


def evaluate(client, rows, checkpoint, harness, config, output):
    output = Path(output)
    results = []
    for repeat in range(config["eval_repetitions"]):
        for row in rows:
            iid = row["metadata"]["instance_id"]
            seed = int(digest([config["seed"], iid, repeat])[:8], 16) % 2147483647
            r = rollout(client, row, checkpoint, harness, config, output / f"r{repeat}" / iid, seed)
            r["repetition"] = repeat
            append(output / "results.jsonl", r)
            results.append(r)
    return results


def mean_score(rows):
    if not rows or any(r["score"] is None for r in rows):
        raise ValueError("Cannot score empty or infrastructure-failed evaluations")
    return sum(r["score"] for r in rows) / len(rows)
