"""Execute and score tasks with the same full mini agent and owned policy server."""

from pathlib import Path

from .data import public_task
from .harness import revision
from .records import append, digest, write
from .reward import grade
from .runtime import run_agent
from .sandbox import open_task
from .serving import serve as serve
from .serving import serve_command as serve_command


def rollout(client, row, checkpoint, harness, config, output, seed):
    output = Path(output)
    c = {**config["runtime"], "sampling_seed": seed}
    sandbox_id = None
    try:
        with open_task(row, c["sandbox"]) as sandbox:
            sandbox_id = sandbox.sandbox.sandbox_id
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
                "sandbox_id": sandbox_id,
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
