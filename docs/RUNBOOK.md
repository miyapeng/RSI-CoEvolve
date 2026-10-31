# Runbook — local assets, Conda, explicit execution

## 1. Existing CPU environment

```bash
cd /user/miyapeng/rsi_evolve/self-hpo-baseline
source /user/miyapeng/miniconda3/etc/profile.d/conda.sh
conda activate self-hpo-baseline
export PYTHONNOUSERSITE=1 MSWEA_SILENT_STARTUP=1
umask 077
python -m pip check
python -m pytest tests/baseline -q
# Includes unmodified native Slime adapter/trajectory regression tests:
bash scripts/baseline/test_cpu.sh
python -m baseline preflight --output runs/preflight-local
```

This dedicated environment was cloned from the existing `harness-policy` Conda environment. The old environment remains separate. Mini-SWE-agent is editable from **this worktree**; Tencent transport uses pinned `e2b==2.62.0`. Orchard is not required by the active baseline. It is a CPU environment, not a ready GPU image. Preflight writes missing prerequisites without printing secrets. `--execute` commands stop when prerequisites are missing; bare commands only save plans.

For a **new machine only**, with the asset links restored, first run `python scripts/baseline/fetch_dependencies.py --ssh --with-final-eval` for missing pinned source dependencies:

```bash
conda env create -f environment-baseline.yml
conda activate self-hpo-baseline
python -m pip install torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip check
```

Do not run the CPU-wheel command in a GPU environment. There is no venv/uv workflow in this baseline. Full installed package versions from the actual local validation are in `runs/validation/packages.json`; the YAML pins the direct CPU dependencies, not a universal CUDA lockfile.

`data` and `models` are ignored symlinks to the existing prototype's asset directories. The 9B link ultimately resolves to `/user/lumingcong/models/Qwen3.5-9B`. Never write to that shared source. The split manifest uses a repository-relative `data/...` path and validates the original raw SHA. Asset paths can therefore change through symlinks without changing task membership.

```bash
python -m baseline split --output runs/splits-check.json
# A plan only; keep execution output separate from this plan directory.
python -m baseline p-step --output runs/p-plan-local --limit 16
```

## 2. Real execution prerequisites

The development allocation has **zero visible GPUs**, CPU-only torch and no Docker/Kubernetes client. Separately, a real single-A100 Cybertron job has now loaded Qwen3.5-9B and passed the shared SGLang tool-call smoke. See [GPU environment validation](GPU_ENVIRONMENT.md) for the image, commands and remaining training dependencies; no GRPO update has run. Real API/Tencent task smoke also works with the discovered public endpoint/template and one existing Scale-SWE mirror; keys are injected temporarily, not stored. See [the small live test](LIVE_SMOKE.md) for those outcomes. The active baseline uses Tencent AGS/E2B for every task and grader. Orchard is not a runtime prerequisite.

Use an authorized GPU allocation with pinned Slime's compatible stack. The existing `v5-miles` environment was inspected but is missing SGLang and is not a verified Slime environment. Do not upgrade it in place.

The authoritative image recipe is `../references/slime/docker/Dockerfile` **at the pinned Slime commit**, not a floating `latest` image. Its defaults include SGLang `v0.5.15.post1-cu129`, Megatron commit `4b4acac9a1d28ea6829c8d4f566d75698a21249d`, Transformer Engine 2.18.0 and NCCL 2.30.7 plus Slime's patches. The existing `ptdbench-slime-v0.3.2-cu129` enterprise image is already synchronized to Langfang and passed actual 9B serving. Its other versions differ from this reference recipe; do not assume it is GRPO-ready. No new image was built or transferred. The initial GPU probes used image Python. The dedicated `self-hpo-gpu` Conda prefix now preserves that tested stack and installs the pinned local agent/Slime/Megatron sources. See [GPU Conda preparation and cluster routing](GPU_CONDA.md). Do not activate the CPU Conda environment in a GPU job.

Install the local mini-SWE-agent and pinned E2B dependencies and baseline into that environment, preserving the image's Torch/CUDA/Slime stack and `swebench==4.1.0`. Ensure `ray`, `sglang`, `megatron-core`, `transformer-engine`, `tensorboard`, `transformers` and `safetensors` import successfully. Native Slime must import from `../references/slime`; the launcher sets its path explicitly and verifies its clean commit.

For distributed training, set `BASELINE_SECRET_ENV_FILE` to an owner-only dotenv file containing `E2B_API_KEY` and `LLM_CENTER_API_KEY`; pass its path, not secret values, to Ray. Set nonsecret `E2B_DOMAIN` and `E2B_TEMPLATE` in the job environment. In proxy mode also export the configured proxy variable (default `http_proxy`); for a dedicated link select `connection: direct`. Prepare `configs/tencent_images.local.yaml` with exact source image / mirrored Tencent tag / expected digest mappings. Do not put registry credentials in this file. Host `skopeo` verifies each tag before creation. Agent and grader use fresh, separate sandboxes. Template network enforcement and actual task-image compatibility still need live validation. Inventory generation, smoke prerequisites and cleanup details are in [TENCENT_SANDBOX.md](TENCENT_SANDBOX.md).

For the default HPO-style proposer set `PROPOSER_MODEL` to the LLMCenter **accessId** and inject `LLM_CENTER_API_KEY` securely. No key is in project files. API base URL, protocol, timeout and proxy inheritance are configurable. The tested Opus accessId requires `anthropic_messages`; use `chat_completions` only for a model exposing that protocol. `proposer_mode: self` uses the currently served policy and must be named as a Self-HPO-style experiment. Neither mode trains a separate API model.

Copy `configs/baseline_9b.yaml` to a job-specific config, adjust GPUs/TP/context/budgets once, and keep it fixed for resume. Defaults are a two-GPU debugging budget, not a measured training memory guarantee. The measured serving smoke used TP=1, context=4096, memory fraction=0.65 on A100 80GB. On Cybertron set `training.checkpoint_root: /data/checkpoints/self-hpo-baseline`; place run records under `/data/results`, logs under the task log mounts, and keep model/data sources read-only. Bounded cctl GPU probes were submitted to `myp-test-2` / `langfang_train` / `ocr-train`. Full training remains unverified.

## 3. Ordered live smoke commands

The commands below use the repository config by default. Add `--config /absolute/path/baseline-job.yaml` consistently if using a job config. All `--execute` commands below consume real resources; run them sequentially only after preflight passes.

```bash
python -m baseline preflight --output runs/preflight-gpu
# Trusted grader only; empty, invalid and reference patches from ONE optimization task.
# The policy never receives the reference patch. Empty-patch zero is a rule, not a test execution.
python -m baseline oracle --limit 1 --output runs/oracle-001 --execute
# Actual 9B server, tool-call check, agent interaction, patch, separate real grader.
python -m baseline rollout --limit 1 --output runs/task-001 --execute
# Fixed seed harness; one configured GRPO step, save, dense export, reload and real task rollout.
python -m baseline p-step --limit 16 --output runs/p-smoke-001 --execute
# Initial frozen policy; two configured candidates, optimization diagnostics and dev comparison.
python -m baseline h-step --limit 8 --output runs/h-smoke-001 --execute
```

`--limit` sets the selected task-pool sizes; it does not change GRPO samples-per-prompt or number of optimizer steps. Those live in the single YAML. Runtime budgets include max steps, wall seconds, per-tool timeout, context/output lengths and sampling parameters. A P-step with no measured effective learning signal is preserved as a failed validation, not published as a fictitious update. A score decrease after a valid update remains a legitimate experiment result.

The original `scripts/p_step/run_p_step.sh`, `prepare_ckpt.sh`, `export_policy.sh` and `scripts/eval/serve_model.sh` are historical 35B/B200 scripts. Do **not** invoke them for this 9B baseline. Dense export uses native `--save-hf` plus frozen auxiliary tensor carryover; never expert fusion.

## 4. Two rounds, baseline switches and resume

```bash
# n_outer_rounds: 2 in the config.
python -m baseline run --mode hpo --output runs/hpo-001 --execute
python -m baseline report --output runs/hpo-001
# Same exact config/mode/limit arguments as the original invocation.
python -m baseline resume --mode hpo --output runs/hpo-001 --execute

# Separate experiments through the SAME loop and trainer.
python -m baseline run --mode h-only --output runs/h-only-001 --execute
python -m baseline run --mode p-only --output runs/p-only-001 --execute
python -m baseline run --mode base --output runs/base-001 --execute
```

Round two reads round one's published checkpoint/harness pair. Each owned policy server is stopped before training; evaluation always starts a fresh server for the selected checkpoint. An occupied port causes failure rather than reuse of an unknown server. Completed H/P/evaluation stages are resumed from disk, including the narrow completed-export-before-publication window. Incomplete optimizer steps/candidate batches are retained and rerun only upon an explicit resume; this is not mid-step distributed optimizer recovery. Configuration/code/data/accepted asset changes invalidate resume. No process-wide `pkill` or shared Docker restart is used.

Default `cross_eval: true` collects four cells on the same small dev subset each round. Standalone replay accepts any saved pair references:

```bash
python -m baseline cross-eval --limit 8 --output runs/cross-001 \
  --old-policy models/Qwen3.5-9B \
  --new-policy /absolute/path/to/exported/policy-0 \
  --old-harness runs/hpo-001/rounds/round_000/harness_before \
  --new-harness runs/hpo-001/rounds/round_000/harness_after --execute
```

Choose audit-holdout checkpoints/times **before** inspecting its outcomes. Put the chosen checkpoint/harness in a separate audit config, then use `python -m baseline audit-holdout --config ... --limit 8 --output runs/audit-001 --execute`. Do not return audit tasks/results to the proposer or use them to choose candidates, stopping rules or hyperparameters.

## 5. Separate final evaluation preparation

These plans were validated offline against existing local dataset hashes: Verified 500, Multilingual 300, **Pro V2 642**. No benchmark execution or score has occurred. The adapters use the same agent generation function, Self-HPO's existing verifiers with an explicit Tencent environment injection, plus pinned official Pro V2 tests. They are outside the H/P loop.

```bash
python -m baseline.final_eval --benchmark verified --output runs/final-verified-plan
python -m baseline.final_eval --benchmark multilingual --output runs/final-multi-plan
python -m baseline.final_eval --benchmark pro-v2 --output runs/final-pro-v2-plan
```

Only after the baseline is reliable and the selected policy/harness are frozen, use the final config and a **new output directory** with `--execute`. Do not repeatedly run these benchmarks to tune the searcher. Dataset commits, task counts, prompt specialization, time budgets and Pro V2 verifier commit are saved in each plan; source-to-Tencent image mappings and expected registry digests are frozen in the resolved config; actual provider runtime digests remain unknown, not invented. Live verifier/image compatibility remains unverified.

## 6. Evidence and expected files

An experiment saves `config.resolved.yaml`, `environment.json`, `splits.json`, `state.json`; each round saves before/after harness snapshots and policy references, H-step candidate history/diagnostics/rollouts, P-step request/logs/token provenance/checkpoints/reload evidence, four-cell evaluations and metrics. `summaries/` contains three CSVs plus a Markdown report. Exceptions preserve their error type and the previous accepted pair; infrastructure failures carry null scores. No generated mock learning curves are used as results.

Local validation evidence and exact known upstream failures: [STATUS.md](STATUS.md). A small passing CPU fixture validates a software contract, not GPU execution, parser correctness, real reward or policy learning.

### Fixed small task subsets

`configs/smoke_a100.yaml` uses one explicitly recorded optimization task and one
separate dev task for bounded engineering validation, with 40 agent steps, 32K
context and 4 A100 GPUs for one GRPO update per P-step. These tiny subsets are
not a representative research evaluation. `task_ids.optimization_pool` and
`task_ids.dev_eval` can select IDs already in the frozen manifest; duplicates,
wrong-split IDs and budgets exceeding the listed subset are rejected. The same
selection is used by preflight, H-step, P-step, cross-eval and outer-loop resume.
Omit `task_ids` to use deterministic seeded subsets from the full training pool.
