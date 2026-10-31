"""Thin native Slime GRPO launcher and dense HF export checks."""

import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from .checkpoint import complete_dense_export, inspect
from .harness import revision, snapshot
from .records import ROOT, SLIME_COMMIT, append, digest, read, sha, write


def check_slime(root):
    root = Path(root)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if head != SLIME_COMMIT or dirty:
        raise ValueError("Expected clean pinned Slime v0.4.0 dependency")


def command(request):
    config, out = request["config"], Path(request["output"])
    t, r, root = config["training"], config["runtime"], Path(config["slime_root"])
    if r["top_p"] != 1.0:
        raise ValueError(
            "Native agent capture has sampled-token logprobs, not nucleus support IDs. "
            "Pinned Slime requires those IDs for top_p<1; set runtime.top_p=1.0 for all H/P/eval phases."
        )
    check_slime(root)
    if (
        revision(request["harness"]["path"]) != request["harness"]["revision"]
        or sha(request["train"]) != request["train_sha256"]
    ):
        raise ValueError("Frozen training harness/data changed")
    model_args = (
        subprocess.check_output(
            [
                "bash",
                "-c",
                'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"',
                "model-args",
                str(root / "scripts/models/qwen3.5-9B.sh"),
            ]
        )
        .decode()
        .rstrip("\0")
        .split("\0")
    )
    batch, samples = t["rollout_batch_size"], t["samples_per_prompt"]
    if batch < 1 or samples < 2 or t["gpus"] % t["tensor_parallel"]:
        raise ValueError("Invalid GRPO batch or dense tensor parallel configuration")
    pairs = {
        "--hf-checkpoint": request["source_checkpoint"],
        "--load": request["source_checkpoint"],
        "--ref-load": request["source_checkpoint"],
        "--save": request["checkpoint_dir"],
        "--save-interval": config["p_training_steps"],
        "--save-hf": str(Path(request["checkpoint_dir"]) / "policy-{rollout_id}"),
        "--prompt-data": request["train"],
        "--input-key": "prompt",
        "--metadata-key": "metadata",
        "--rollout-function-path": "baseline.slime_rollout.generate_rollout",
        "--custom-generate-function-path": "baseline.slime_rollout.generate",
        "--custom-reward-post-process-path": "baseline.slime_rollout.post_process_rewards",
        "--num-rollout": config["p_training_steps"],
        "--rollout-batch-size": batch,
        "--n-samples-per-prompt": samples,
        "--global-batch-size": batch * samples,
        "--rollout-max-response-len": r["max_tokens"],
        "--rollout-max-context-len": r["max_context_tokens"],
        "--rollout-temperature": r["temperature"],
        "--rollout-top-p": r["top_p"],
        "--actor-num-nodes": 1,
        "--actor-num-gpus-per-node": t["gpus"],
        "--tensor-model-parallel-size": t["tensor_parallel"],
        "--pipeline-model-parallel-size": 1,
        "--context-parallel-size": 1,
        "--recompute-granularity": "full",
        "--recompute-method": "uniform",
        "--recompute-num-layers": 1,
        "--max-tokens-per-gpu": t["max_tokens_per_gpu"],
        "--log-probs-chunk-size": t.get("log_probs_chunk_size", 512),
        "--advantage-estimator": "grpo",
        "--kl-loss-coef": 0.0,
        "--kl-loss-type": "low_var_kl",
        "--entropy-coef": 0.0,
        "--eps-clip": 0.2,
        "--eps-clip-high": 0.28,
        "--optimizer": "adam",
        "--lr": t["learning_rate"],
        "--lr-decay-style": "constant",
        "--weight-decay": 0.1,
        "--adam-beta1": 0.9,
        "--adam-beta2": 0.98,
        "--rollout-num-gpus-per-engine": t["rollout_gpus_per_engine"],
        "--sglang-mem-fraction-static": config["serving"]["mem_fraction_static"],
        "--sglang-context-length": r["max_context_tokens"],
        "--sglang-attention-backend": config["serving"]["attention_backend"],
        "--sglang-tool-call-parser": "qwen3_coder",
        "--sglang-reasoning-parser": "qwen3",
        "--sglang-mamba-scheduler-strategy": "extra_buffer",
        "--attention-dropout": 0.0,
        "--hidden-dropout": 0.0,
        "--attention-backend": "flash",
        "--rollout-data-transport": "object-store",
        "--seed": config["seed"],
        "--tb-project-name": "self-hpo-baseline",
        "--tb-experiment-name": out.name,
    }
    flags = [
        "--colocate",
        "--sequence-parallel",
        "--use-dynamic-batch-size",
        "--use-kl-loss",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--use-tensorboard",
    ]
    if not t.get("gradient_accumulation_fusion", True):
        flags.append("--no-gradient-accumulation-fusion")
    if t.get("recompute_loss", False):
        flags.append("--recompute-loss-function")
    return [
        sys.executable,
        str(root / "train.py"),
        *model_args,
        *flags,
        *(v for k, value in pairs.items() for v in (k, str(value))),
    ]


def prepare(checkpoint, harness, rows, config, output):
    output = Path(output).resolve()
    if (output / "request.json").exists():
        raise ValueError("P-step output already exists; resume completed stages or use a new experiment")
    target = snapshot(harness, output / "harness")
    for row in rows:
        if row["split"] != "optimization_pool":
            raise ValueError("P-step only trains on optimization_pool")
        append(
            output / "train.jsonl",
            {
                # Qwen3.5 loads a multimodal Processor even for text-only tasks.
                # Slime requires conversation form; actual model messages are
                # still rendered by the shared agent through OpenAIAdapter.
                "prompt": [{"role": "user", "content": row["problem_statement"]}],
                "metadata": {"task_id": row["metadata"]["instance_id"], "split": row["split"]},
            },
        )
    request = {
        "source_checkpoint": checkpoint,
        "harness": target,
        "config": config,
        "output": str(output),
        "train": str(output / "train.jsonl"),
        "train_sha256": sha(output / "train.jsonl"),
        "manifest_sha256": sha(config["dataset_split_manifest"]),
        "checkpoint_dir": str(
            Path(config["training"]["checkpoint_root"]) / (output.parent.name + "-" + digest(str(output))[:12])
        ),
    }
    write(output / "request.json", request)
    write(
        output / "launch-plan.json",
        {
            "status": "NOT_RUN",
            "command": command(request),
            "harness_revision": target["revision"],
            "dense_hf_export": "native --save-hf; no expert fusion",
            "final_benchmarks_in_training": False,
        },
    )
    return request


def collect_statistics(output):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    output = Path(output)
    acc = EventAccumulator(str(output / "tensorboard"), size_guidance={"scalars": 0})
    acc.Reload()
    metrics = {
        tag: [{"step": e.step, "value": e.value} for e in acc.Scalars(tag)]
        for tag in acc.Tags()["scalars"]
        if tag.startswith("train/")
    }
    grads = metrics.get("train/grad_norm", [])
    effective = sum(math.isfinite(g["value"]) and g["value"] > 0 for g in grads)
    groups = defaultdict(list)
    tokens = 0
    for path in (output / "rollout_provenance").glob("*.json"):
        row = read(path)
        groups[row.get("policy_update_iteration"), row["group_index"]].append(row["reward"])
        tokens += row["trainable_tokens"]
    return {
        "training_metrics": metrics,
        "observed_finite_nonzero_gradient_steps": effective,
        "effective_loss_tokens": tokens,
        "reward_groups": len(groups),
        "mixed_reward_group_fraction": sum(len(set(v)) > 1 for v in groups.values()) / len(groups) if groups else None,
    }


def train(checkpoint, harness, rows, config, output):
    request = prepare(checkpoint, harness, rows, config, output)
    output = Path(output)
    before = inspect(checkpoint)
    write(output / "source-checkpoint.json", before)
    with (output / "train.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-m", "baseline.train_worker", str(output / "request.json")],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if revision(request["harness"]["path"]) != request["harness"]["revision"]:
        raise ValueError("Training modified the fixed harness")
    exported = Path(request["checkpoint_dir"]) / f"policy-{config['p_training_steps'] - 1}"
    export_completion = complete_dense_export(checkpoint, exported)
    after = inspect(exported)
    stats = collect_statistics(output)
    stats["changed_tensor_probes"] = sum(
        v != before["tensor_probes"][k]
        for k, v in after["tensor_probes"].items()
        if k.startswith("model.language_model.") or k == "lm_head.weight"
    )
    if (
        not stats["training_metrics"]
        or stats["effective_loss_tokens"] <= 0
        or stats["observed_finite_nonzero_gradient_steps"] <= 0
    ):
        write(output / "training-statistics.json", stats)
        raise RuntimeError("No measured effective learning signal; pair remains unchanged")
    if not stats["changed_tensor_probes"]:
        raise RuntimeError("No parameter changes detected in sparse probes; cannot validate an update")
    # A fresh server + real tool call checks export/reload. It is closed before the next evaluation/round.
    from .execution import evaluate, serve

    with serve(str(exported), config, output / "reload") as client:
        reloaded_rollout = evaluate(
            client, rows[:1], str(exported), request["harness"]["path"], config, output / "reload" / "rollout"
        )
    result = {
        "checkpoint": str(exported),
        "harness_revision": request["harness"]["revision"],
        "statistics": stats,
        "checkpoint_inspection": after,
        "export_completion": export_completion,
        "reload_verified": True,
        "reload_rollout": reloaded_rollout,
        "simulated": False,
    }
    write(output / "result.json", result)
    return result
