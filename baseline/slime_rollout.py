"""Native Slime token/logprob capture around the identical mini runtime used by H-step."""

import asyncio
import json
import os
import uuid
from pathlib import Path

from .data import tasks
from .execution import rollout
from .harness import revision
from .records import read, sha, write


class CaptureService:
    def __init__(self, args):
        from slime.agent.adapters import OpenAIAdapter
        from slime.agent.aiohttp_threaded import run_app_in_thread
        from slime.utils.processing_utils import load_tokenizer

        self.adapter = OpenAIAdapter(
            tokenizer=load_tokenizer(args.hf_checkpoint, trust_remote_code=False),
            sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            tool_parser="qwen3_coder",
            reasoning_parser="qwen3",
            fork_threshold_tokens=0,
        )
        self.handle = run_app_in_thread(
            self.adapter.app, host="127.0.0.1", port=0, runner_kwargs={"handler_cancellation": True}
        )

    async def on_loop(self, fn, *args, **kwargs):
        async def call():
            result = fn(*args, **kwargs)
            return await result if asyncio.iscoroutine(result) else result

        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(call(), self.handle.loop))


_service = None


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    """Attach the synchronous policy-update iteration; delegate the whole rollout to Slime."""
    from slime.rollout.sglang_rollout import generate_rollout as native_rollout

    args.baseline_rollout_iteration = rollout_id
    return native_rollout(args, rollout_id, data_source, evaluation=evaluation)


async def generate(args, sample, sampling_params, evaluation=False):
    global _service
    request = read(os.environ["BASELINE_TRAIN_REQUEST"])
    config = request["config"]
    if evaluation or sample.metadata["split"] != "optimization_pool":
        raise ValueError("This adapter never trains/evaluates on final benchmarks")
    if (
        sha(config["dataset_split_manifest"]) != request["manifest_sha256"]
        or sha(request["train"]) != request["train_sha256"]
    ):
        raise ValueError("Frozen training data changed")
    target = request["harness"]
    if revision(target["path"]) != target["revision"]:
        raise ValueError("Frozen training harness changed")
    allowed = {json.loads(line)["metadata"]["task_id"] for line in Path(request["train"]).read_text().splitlines()}
    iid = sample.metadata["task_id"]
    if iid not in allowed:
        raise ValueError("Task outside frozen P-step selection")
    row = next(
        r for r in tasks(config["dataset_split_manifest"], "optimization_pool") if r["metadata"]["instance_id"] == iid
    )
    if _service is None:
        _service = CaptureService(args)
    service, sid = _service, uuid.uuid4().hex
    await service.on_loop(
        service.adapter.open_session,
        sid,
        sampling_defaults=sampling_params,
        max_context_tokens=config["runtime"]["max_context_tokens"],
    )

    def execute():
        import httpx
        from openai import OpenAI

        c = {
            **config,
            "runtime": {
                **config["runtime"],
                "max_tokens": sampling_params["max_new_tokens"],
                "temperature": sampling_params.get("temperature", config["runtime"]["temperature"]),
            },
        }
        with OpenAI(
            base_url=f"http://127.0.0.1:{service.handle.port}/v1",
            api_key=sid,
            max_retries=0,
            http_client=httpx.Client(trust_env=False),
        ) as client:
            result = rollout(
                client,
                row,
                request["source_checkpoint"],
                target["path"],
                c,
                Path(request["output"]) / "rollouts" / sid,
                config["seed"] + sample.index,
            )
            result["policy_update_iteration"] = getattr(args, "baseline_rollout_iteration", None)
            write(Path(request["output"]) / "rollouts" / sid / "result.json", result)
            return result

    try:
        result = await asyncio.to_thread(execute)
        if revision(target["path"]) != target["revision"]:
            raise ValueError("Harness mutated during rollout")
        samples = await service.on_loop(
            service.adapter.finish_session,
            sid,
            base_sample=sample,
            reward=result["training_reward"],
            extra_metadata={"harness_revision": target["revision"], "task_id": iid},
        )
        if not samples:
            raise ValueError("Rollout produced no trainable sample")
        for s in samples:
            if (
                len(s.loss_mask) != s.response_length
                or len(s.rollout_log_probs) != s.response_length
                or not sum(s.loss_mask)
                or not all(x in (0, 1) for x in s.loss_mask)
            ):
                raise ValueError("Invalid native Slime training token/mask/logprob alignment")
        write(
            Path(request["output"]) / "rollout_provenance" / f"{sid}.json",
            {
                "task_id": iid,
                "group_index": sample.group_index,
                "sample_index": sample.index,
                "policy_checkpoint": request["source_checkpoint"],
                "policy_update_iteration": getattr(args, "baseline_rollout_iteration", None),
                "harness_revision": target["revision"],
                "reward": result["training_reward"],
                "trainable_tokens": sum(sum(s.loss_mask) for s in samples),
                "segments": [
                    {
                        "tokens": s.tokens,
                        "loss_mask": s.loss_mask,
                        "response_length": s.response_length,
                        "rollout_log_probs": s.rollout_log_probs,
                    }
                    for s in samples
                ],
                "simulated": False,
            },
        )
        return samples
    finally:
        await service.on_loop(service.adapter.drop_session, sid, wait_timeout=30)
