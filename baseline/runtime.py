"""One full mini agent for search, dev, RL and final-test generation."""

import copy
import re
import time
from pathlib import Path
from types import SimpleNamespace

from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import LimitsExceeded
from minisweagent.models.utils.actions_toolcall import (
    TOOLS,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.run.benchmarks.utils.runner_common import _NoopProgressManager
from openai import BadRequestError

from .harness import load, revision
from .records import append, write


class Model:
    def __init__(self, client, name, settings, template, trace, deadline):
        self.client, self.name, self.settings = client, name, settings
        self.template, self.trace, self.deadline = template, trace, deadline
        self.tokens = 0
        self.usage_available = True

    def query(self, messages):
        if time.monotonic() >= self.deadline:
            raise LimitsExceeded({"role": "exit", "content": "wall timeout", "extra": {"exit_status": "TimeoutError"}})
        request = {
            "model": self.name,
            "messages": [{k: copy.deepcopy(v) for k, v in m.items() if k != "extra"} for m in messages],
            "tools": TOOLS,
            "max_tokens": self.settings["max_tokens"],
            "temperature": self.settings["temperature"],
            "top_p": self.settings["top_p"],
            "seed": self.settings["sampling_seed"],
        }
        append(self.trace, {"event": "model_input", "request": request})
        try:
            response = self.client.chat.completions.create(
                **request, timeout=max(0.01, min(120, self.deadline - time.monotonic()))
            )
        except BadRequestError as exc:
            body = exc.body if isinstance(exc.body, dict) else {}
            message = body.get("message", "")
            if (
                not isinstance(message, str)
                or "Requested token count exceeds the model's maximum context length" not in message
            ):
                raise
            counts = re.search(
                r"maximum context length of (\d+) tokens.*?(\d+) tokens from the input messages", message
            )
            if not counts:
                raise
            remaining = int(counts[1]) - int(counts[2])
            append(
                self.trace, {"event": "context_limit", "http_status": 400, "message": message, "remaining": remaining}
            )
            if remaining <= 0:
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": "Context budget exhausted",
                        "extra": {"exit_status": "ContextLimitExceeded", "submission": ""},
                    }
                ) from exc
            if remaining >= request["max_tokens"]:
                raise
            # Match native Slime's exact remaining-context cap. The rejected
            # request generated no tokens; only this specific cap is retried once.
            request["max_tokens"] = remaining
            append(self.trace, {"event": "model_input", "request": request})
            response = self.client.chat.completions.create(
                **request, timeout=max(0.01, min(120, self.deadline - time.monotonic()))
            )
        append(self.trace, {"event": "model_output", "response": response.model_dump(mode="json")})
        if response.usage is None:
            self.usage_available = False
        else:
            self.tokens += response.usage.total_tokens
        if (
            response.choices[0].finish_reason == "length"
            and response.usage is not None
            and response.usage.completion_tokens == 0
            and response.usage.prompt_tokens >= self.settings["max_context_tokens"]
        ):
            # Native Slime returns this empty turn when its context budget is exhausted.
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "Context budget exhausted",
                    "extra": {"exit_status": "ContextLimitExceeded", "submission": ""},
                }
            )
        m = response.choices[0].message.model_dump(exclude_none=True)
        m["extra"] = {
            "actions": parse_toolcall_actions(
                response.choices[0].message.tool_calls or [],
                format_error_template=self.template["format_error_template"],
            )
        }
        return m

    def format_message(self, **kw):
        return kw

    def format_observation_messages(self, message, outputs, template_vars=None):
        return format_toolcall_observation_messages(
            actions=message["extra"]["actions"],
            outputs=outputs,
            observation_template=self.template["observation_template"],
            template_vars=template_vars,
        )

    def get_template_vars(self):
        return {"model_name": self.name}

    def serialize(self):
        return {"info": {"model_name": self.name, "tokens": self.tokens if self.usage_available else None}}


class Environment:
    def __init__(self, sandbox, trace, deadline, settings):
        self._sandbox, self._trace, self._deadline, self._settings = sandbox, trace, deadline, settings
        self.config = SimpleNamespace(cwd=sandbox.task["cwd"])
        self.calls = 0

    def execute(self, action, cwd="", timeout=None):
        remaining = int(self._deadline - time.monotonic())
        if remaining <= 0:
            raise LimitsExceeded({"role": "exit", "content": "wall timeout", "extra": {"exit_status": "TimeoutError"}})
        result = self._sandbox.execute(
            action, cwd=cwd, timeout=min(timeout or self._settings["command_timeout"], remaining)
        )
        self.calls += 1
        append(self._trace, {"event": "tool", "action": action, "result": result})
        DockerEnvironment._check_finished(self, result)
        return result

    def get_template_vars(self):
        return {"cwd": self.config.cwd}

    def serialize(self):
        return {"info": {"sandbox_image": self._sandbox.task["image"], "cwd": self.config.cwd}}


def run_agent(client, sandbox, task, harness, checkpoint, config, output):
    """All caller modes use this function, with the same actual AgentHarness class."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected = revision(harness)
    cls, hc = load(harness, expected)
    start = time.monotonic()
    deadline = start + config["max_seconds"]
    model = Model(client, checkpoint, config, hc["model"], output / "events.jsonl", deadline)
    env = Environment(sandbox, output / "events.jsonl", deadline, config)
    agent = cls(
        model,
        env,
        progress_manager=_NoopProgressManager(),
        instance_id=task["task_id"],
        deadline=deadline,
        **{**hc["agent"], "step_limit": config["max_steps"], "cost_limit": 0, "output_path": None},
    )
    try:
        result = agent.run(task["problem_statement"])
    finally:
        write(output / "trajectory.json", agent.serialize())
    if revision(harness) != expected:
        raise ValueError("Harness mutated during execution")
    return {
        "task_id": task["task_id"],
        "dataset_source": task["dataset_source"],
        "split": task["split"],
        "policy_checkpoint": checkpoint,
        "harness_revision": expected,
        "sampling_seed": config["sampling_seed"],
        "sampling_params": {k: config[k] for k in ("temperature", "top_p", "max_tokens", "max_steps", "max_seconds")},
        "sandbox_image": task["image"],
        "sandbox_backend": config["sandbox"]["backend"],
        "execution_image": getattr(sandbox, "execution_image", None),
        "registry_digest_before_create": getattr(sandbox, "registry_digest", None),
        "provider_runtime_digest": None,
        "sandbox_id": sandbox.sandbox.sandbox_id,
        "patch": result.get("submission", ""),
        "termination_reason": result.get("exit_status"),
        "error_type": None,
        "score": None,
        "tokens": model.tokens if model.usage_available else None,
        "tool_calls": env.calls,
        "latency_seconds": time.monotonic() - start,
        "trajectory": str((output / "trajectory.json").resolve()),
        "simulated": False,
    }
