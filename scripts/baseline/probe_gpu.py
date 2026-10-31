"""Bounded image compatibility checks; never substitutes for a GRPO update.

Run with the GPU image's Python, not the CPU development Conda environment.
Each check runs in a fresh process so an import/CUDA failure remains attributable.
No downloads, package installs, task sandboxes or credentials are used.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def check(name, model):
    if name == "versions":
        packages = {}
        for package in (
            "torch",
            "transformers",
            "sglang",
            "sgl-kernel",
            "megatron-core",
            "transformer-engine",
            "flash-attn",
            "flash-linear-attention",
            "ray",
            "slime",
            "e2b",
            "swebench",
            "mini-swe-agent",
        ):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        return {"python": sys.version, "executable": sys.executable, "packages": packages}
    if name == "model_files_and_tokenizer":
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(model, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        messages = [{"role": "user", "content": "Inspect the repository using bash."}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert "bash" in rendered and "Inspect the repository" in rendered
        index = json.loads((Path(model) / "model.safetensors.index.json").read_text())
        shards = sorted(set(index["weight_map"].values()))
        for shard in shards:
            with (Path(model) / shard).open("rb") as stream:
                assert len(stream.read(8)) == 8, shard
        return {
            "model_path": str(Path(model).resolve()),
            "model_type": config.model_type,
            "shards_readable": len(shards),
            "template_chars": len(rendered),
        }
    if name == "slime_qwen_imports":
        modules = {}
        for module in (
            "slime",
            "slime.backends.megatron_utils.arguments",
            "slime_plugins.models.qwen3_5",
            "sglang.srt.models.qwen3_5",
        ):
            modules[module] = importlib.import_module(module).__file__
        return modules

    import torch

    assert torch.cuda.is_available(), "CUDA is unavailable in this Python environment"
    torch.manual_seed(42)
    if name == "torch_backward":
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        (x @ x.T).float().square().mean().backward()
        assert torch.isfinite(x.grad).all().item()
        torch.cuda.synchronize()
        return {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "visible_gpus": torch.cuda.device_count(),
            "nccl": torch.cuda.nccl.version(),
        }
    if name == "transformer_engine_backward":
        import transformer_engine.pytorch as te

        layer = te.Linear(128, 128, params_dtype=torch.bfloat16, device="cuda")
        x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        layer(x).float().square().mean().backward()
        assert torch.isfinite(x.grad).all().item()
        assert torch.isfinite(layer.weight.grad).all().item()
    elif name == "flash_attention_backward":
        from flash_attn import flash_attn_func

        q, k, v = [
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
        ]
        flash_attn_func(q, k, v, causal=True).float().square().mean().backward()
        assert all(torch.isfinite(x.grad).all().item() for x in (q, k, v))
    elif name == "qwen_delta_backward":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        q, k, v = [
            torch.randn(1, 64, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
        ]
        g = -torch.rand(1, 64, 2, device="cuda", requires_grad=True)
        beta = torch.rand(1, 64, 2, device="cuda", requires_grad=True)
        out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        out.float().square().mean().backward()
        assert all(torch.isfinite(x.grad).all().item() for x in (q, k, v, beta))
    elif name == "sgl_kernel":
        import sgl_kernel

        x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
        result = sgl_kernel.rmsnorm(x, weight, 1e-6)
        reference = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(result.float(), reference, atol=0.03, rtol=0.03)
    else:
        raise ValueError(name)
    torch.cuda.synchronize()
    return {"finite_forward_backward": True} if name.endswith("backward") else {"matches_reference": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check")
    parser.add_argument("--check-timeout", type=int, default=120)
    args = parser.parse_args()
    if args.check:
        result = check(args.check, args.model)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"job_id": os.environ.get("JOB_ID"), "simulated": False, "grpo_update": "NOT_RUN", "checks": {}}
    checks = (
        "versions",
        "torch_backward",
        "model_files_and_tokenizer",
        "transformer_engine_backward",
        "flash_attention_backward",
        "qwen_delta_backward",
        "sgl_kernel",
        "slime_qwen_imports",
    )
    for name in checks:
        t0 = time.monotonic()
        result_path = args.output / f"{name}.json"
        command = [sys.executable, __file__, "--model", args.model, "--check", name, "--output", str(result_path)]
        with (args.output / f"{name}.log").open("w") as log:
            try:
                proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.check_timeout)
                item = {"passed": proc.returncode == 0, "returncode": proc.returncode}
            except subprocess.TimeoutExpired:
                item = {"passed": False, "error": "check_timeout"}
        item["seconds"] = round(time.monotonic() - t0, 3)
        if item["passed"]:
            item["result"] = json.loads(result_path.read_text())
        report["checks"][name] = item
        report["seconds"] = round(time.monotonic() - started, 3)
        report["complete"] = len(report["checks"]) == len(checks)
        report["passed"] = report["complete"] and all(c["passed"] for c in report["checks"].values())
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"check": name, **item}), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
