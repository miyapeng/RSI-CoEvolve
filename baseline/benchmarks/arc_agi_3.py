"""Official ARC Prize BenchmarkingAgent and online scorecard, unchanged game actions."""

import contextlib
import importlib
import logging
import os
from pathlib import Path

import yaml

from baseline.records import digest, sha, write

from .base import TaskInfo
from .official import OfficialBenchmark, OfficialTask, path_value, safe_id


def profile_path(options):
    return (
        path_value(options["model_config_path"])
        if options.get("model_config_path")
        else path_value(options["repo"]) / "benchmarking/model_configs.yaml"
    )


def model_profile(options):
    """Read configuration only; native schema validation still runs in the worker."""
    entries = yaml.safe_load(profile_path(options).read_text())
    matches = [e for e in entries if e["id"] == options["model_config_id"]]
    if len(matches) != 1:
        raise ValueError("Expected one matching ARC model profile")

    def check_credentials(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in {"api_key", "authorization", "access_token", "password"}:
                    raise ValueError("ARC profiles must reference credential environment variables")
                check_credentials(child)
        elif isinstance(value, list):
            for child in value:
                check_credentials(child)

    check_credentials(matches[0])
    return matches[0]


@contextlib.contextmanager
def native_settings(request):
    """Configure the native loader in this worker, without replacing any behavior."""
    model_config = importlib.import_module("benchmarking.model_config")

    options = request["options"]
    path = profile_path(options)
    if sha(path) != options["model_config_sha256"]:
        raise ValueError("Frozen ARC model profile changed")
    cap = options.get("max_actions_hard_cap")
    if os.environ.get("MAX_ACTIONS_HARD_CAP"):
        raise ValueError("Unset MAX_ACTIONS_HARD_CAP; declare the smoke budget in config")
    if cap is not None and (not options.get("smoke_test") or type(cap) is not int or cap < 2):
        raise ValueError("ARC action cap requires an explicit smoke_test and at least two actions")
    saved = dict(os.environ)
    original = getattr(model_config, "MODEL_CONFIG_PATH", None)
    try:
        if options.get("model_config_path"):
            model_config.MODEL_CONFIG_PATH = path
        if options.get("anthropic_base_url"):
            os.environ["ANTHROPIC_BASE_URL"] = options["anthropic_base_url"]
        elif os.environ.get("ANTHROPIC_BASE_URL"):
            raise ValueError("Declare ANTHROPIC_BASE_URL as official.anthropic_base_url")
        if cap is not None:
            os.environ["MAX_ACTIONS_HARD_CAP"] = str(cap)
        yield
    finally:
        if options.get("model_config_path"):
            model_config.MODEL_CONFIG_PATH = original
        os.environ.clear()
        os.environ.update(saved)


class SecretFilter(logging.Filter):
    """The upstream Swarm logs its full scorecard, including the ARC API key."""

    def __init__(self, values):
        super().__init__()
        self.values = [v for v in values if v]

    def filter(self, record):
        message = record.getMessage()
        for value in self.values:
            message = message.replace(value, "[REDACTED]")
        record.msg, record.args = message, ()
        return True


class ARCAGI3(OfficialBenchmark):
    name = "arc_agi_3"
    url = "https://github.com/arcprize/arc-agi-3-benchmarking"
    commit = "353d38be471395b5ae47dbb44df95440677397e1"

    def plan(self, task, checkpoint, harness, config, output, seed):
        options = dict(config["official"])
        if options.get("model_config_path"):
            options["model_config_path"] = str(profile_path(options))
        options["model_config_sha256"] = sha(profile_path(options))
        if model_profile(options)["request"]["model"] != checkpoint:
            raise ValueError("ARC model profile and configured policy differ")
        return super().plan(task, checkpoint, harness, {**config, "official": options}, output, seed)

    def discover(self, config):
        # No API calls at discovery. Explicit full game IDs, not prefix expansion to all games.
        rows = []
        for iid in config["official"].get("task_ids", []):
            safe_id(iid)
            if "/" in iid:
                raise ValueError("ARC game ID must be a single identifier")
            rows.append(
                OfficialTask(
                    TaskInfo(iid, "arcprize/ARC-AGI-3", "final_test"),
                    {"task_id": iid, "observations": []},
                    {"game_id": iid},
                    digest(iid),
                )
            )
        return rows


def run_native(request):
    with native_settings(request):
        return _run_native(request)


def _run_native(request):
    options = request["options"]
    root_url = options["arc_base_url"].rstrip("/")
    os.environ["ARC_BASE_URL"] = root_url

    from benchmarking.base import ExitReason
    from benchmarking.model_config import get_model_config
    from benchmarking.runtime_clients import validate_model_credentials
    from benchmarking.swarm import Swarm

    entry = get_model_config(options["model_config_id"])
    if entry["request"]["model"] != request["model"]:
        raise ValueError("ARC official model profile and recorded model differ")
    validate_model_credentials(
        runtime_config=entry["runtime"], client_config=entry["client"], config_id=options["model_config_id"]
    )
    if not os.environ.get("ARC_API_KEY"):
        raise ValueError("ARC_API_KEY is required by the official online harness")
    out = Path(request["output"]) / "native"
    out.mkdir()
    os.environ["RECORDINGS_DIR"] = str(out)
    # No prompt/parser/agent/score formula changes. The explicit development cap
    # uses upstream's own MAX_ACTIONS_HARD_CAP and never masquerades as a full run.
    secret_filter = SecretFilter([os.environ["ARC_API_KEY"], os.environ.get(entry["client"].get("api_key_env", ""))])
    handlers = list(logging.getLogger().handlers)
    for handler in handlers:
        handler.addFilter(secret_filter)
    swarm = None
    try:
        swarm = Swarm(
            root_url,
            [request["task"]["game_id"]],
            config=options["model_config_id"],
            tags=["RSI-CoEvolve", "smoke" if options.get("smoke_test") else "evaluation"],
        )
        card = swarm.main()
    finally:
        try:
            if swarm is not None:
                swarm.request_shutdown()
                swarm.close_scorecard()
                swarm.cleanup()
        finally:
            for handler in handlers:
                handler.removeFilter(secret_filter)
    if card is None:
        raise RuntimeError("Official ARC scorecard was not returned")
    card_data = card.model_dump(mode="json", exclude={"api_key"})
    write(out / "scorecard.json", card_data)
    if len(swarm.agents) != 1:
        raise RuntimeError("Expected exactly one ARC game")
    agent = swarm.agents[0]
    # Upstream threads can fail without raising in Swarm.main(); never count that as a valid score.
    if agent.exit_reason not in {ExitReason.GAME_WIN, ExitReason.ACTION_BUDGET, ExitReason.TIME_BUDGET}:
        raise RuntimeError(f"ARC agent did not finish normally: {agent.exit_reason}")
    return {
        "task_id": request["task_id"],
        "score": card_data["score"],
        "metrics": {
            "official_scorecard": card_data,
            "model_config": entry,
            "levels_completed": agent.levels_completed,
            "smoke_test": options.get("smoke_test", False),
            "max_actions_hard_cap": options.get("max_actions_hard_cap"),
            # Native per-step files use cwd/recordings, independently of RECORDINGS_DIR.
            "step_records": str((Path(request["output"]) / agent.run_dir).resolve()),
            "cost_usd": agent.run_record.total_usage.cost if entry.get("pricing") else None,
            "cost_source": "profile_estimate" if entry.get("pricing") else "unknown_no_pricing",
        },
        "termination_reason": agent.exit_reason.value,
        "tokens": agent.token_counter,
        "tool_calls": agent.action_counter,
    }
