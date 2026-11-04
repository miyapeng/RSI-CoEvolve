"""Official ARC Prize BenchmarkingAgent and online scorecard, unchanged game actions."""

import os
from pathlib import Path

from baseline.records import digest, write

from .base import TaskInfo
from .official import OfficialBenchmark, OfficialTask, safe_id


class ARCAGI3(OfficialBenchmark):
    name = "arc_agi_3"
    url = "https://github.com/arcprize/arc-agi-3-benchmarking"
    commit = "353d38be471395b5ae47dbb44df95440677397e1"

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
    # Avoid implicit experiment budget overrides inherited from another shell run.
    if os.environ.get("MAX_ACTIONS_HARD_CAP"):
        raise ValueError("Unset MAX_ACTIONS_HARD_CAP to retain the pinned official profile budget")
    swarm = Swarm(root_url, [request["task"]["game_id"]], config=options["model_config_id"], tags=["RSI-CoEvolve"])
    try:
        card = swarm.main()
    finally:
        swarm.request_shutdown()
        swarm.close_scorecard()
        swarm.cleanup()
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
        "metrics": {"official_scorecard": card_data, "model_config": entry, "levels_completed": agent.levels_completed},
        "termination_reason": agent.exit_reason.value,
        "tokens": agent.token_counter,
        "tool_calls": agent.action_counter,
    }
