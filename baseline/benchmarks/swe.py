"""Existing Self-HPO SWE tasks, mini agent, Tencent sandbox and original reward."""

from pathlib import Path

from ..records import write
from .base import TaskInfo, Verification


class SWEBenchmark:
    name = "swe"

    def load_tasks(self, config, role, limit):
        """Select a recorded small subset without moving tasks between frozen splits."""
        from ..data import tasks

        ids = config.get("task_ids", {}).get(role)
        if ids is None:
            return tasks(config["dataset_split_manifest"], role, limit)
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError("task_ids must be a nonempty list of unique task IDs")
        available = {r["metadata"]["instance_id"]: r for r in tasks(config["dataset_split_manifest"], role)}
        if set(ids) - available.keys() or limit > len(ids):
            raise ValueError(f"Requested task subset/budget is outside {role}")
        return [available[i] for i in ids[:limit]]

    def task_info(self, row):
        return TaskInfo(row["metadata"]["instance_id"], row["dataset_source"], row["split"])

    def public_task(self, row):
        from ..data import public_task

        # Preserve the exact legacy projection consumed by mini-SWE-agent.
        # Its `image` is a container image, never a visual observation.
        return public_task(row)

    def verify(self, row, episode, runtime_config, output) -> Verification:
        from ..reward import grade

        details = grade(row, episode["patch"], runtime_config, output)
        return {"score": details["score"], "training_reward": 1.0 if details["resolved"] else -1.0, "details": details}

    def run_episode(self, client, row, checkpoint, harness, config, output, seed):
        from ..harness import revision
        from ..runtime import run_agent
        from ..sandbox import open_task

        output = Path(output)
        c = {**config["runtime"], "sampling_seed": seed}
        sandbox_id = None
        try:
            with open_task(row, c["sandbox"]) as sandbox:
                sandbox_id = sandbox.sandbox.sandbox_id
                write(output / "sandbox.json", {"sandbox_id": sandbox_id, "task_id": row["metadata"]["instance_id"]})
                result = run_agent(client, sandbox, self.public_task(row), harness, checkpoint, c, output)
            verified = self.verify(row, result, c, output / "grading")
            result.update(
                score=verified["score"], training_reward=verified["training_reward"], reward=verified["details"]
            )
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
