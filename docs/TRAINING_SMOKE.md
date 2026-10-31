# Bounded real training smoke

Use the existing registered image plus the dedicated **GPU Conda** environment,
not the CPU development environment. See [GPU_CONDA.md](GPU_CONDA.md) for its
actual preparation and source pins. No source model or CUDA package redownload
is needed on this workspace.

`configs/smoke_a100.yaml` fixes the small engineering budget: one optimization
task, one independent dev task, 4 trajectories per GRPO group, one native Slime
update per P-step, 4 A100 80GB GPUs (TP4 / DP1), 40 agent steps and 32K context.
Native logprob chunks of 512 and loss recomputation bound training memory.
All H/P/eval sampling uses top_p=1 because the pinned native Agent Adapter does
not capture the nucleus support required for top_p<1 training. Larger studies
must choose their budget before observing results; this smoke is not a benchmark.

## Submit the one-update check

The private files below already exist on this workspace. They are owner-readable,
outside Git, and are loaded by workers without putting keys in Ray/job metadata.
The command is for the currently verified Langfang public-direct Tencent route.

```bash
cd /user/miyapeng/rsi_evolve/self-hpo-baseline
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CCTL_NO_UPDATE_CHECK=1 \
  cctl job create \
    --project myp-test-2 --billing-account-id N00002 \
    --cluster langfang_train --resource-pool ocr-train \
    --image ptdbench-slime-v0.3.2-cu129 --priority PREEMPTABLE \
    --gpu 4 --gpu-model a100 --cpu 16 --memory 384 \
    --entry 'bash /user/miyapeng/rsi_evolve/self-hpo-baseline/scripts/baseline/run_training_smoke.sh' \
    --env BASELINE_PHASE=p-step \
    --env BASELINE_CONFIG=/user/miyapeng/rsi_evolve/self-hpo-baseline/configs/smoke_a100.yaml \
    --env GPU_CONDA_PREFIX=/user/miyapeng/miniconda3/envs/self-hpo-gpu \
    --env BASELINE_TIMEOUT=3600 \
    --env BASELINE_SECRET_ENV_FILE=/user/miyapeng/.secrets/rsi-coevolve.env \
    --env REGISTRY_AUTH_FILE=/user/miyapeng/.secrets/rsi-registry-auth.json \
    --env E2B_DOMAIN=ap-beijing.tencentags.com \
    --env E2B_TEMPLATE=code-agent-cfs2-new \
    --env NO_PROXY=modelbest.tencentcloudcr.com,.tencentags.com
```

On an already allocated compatible GPU job, activate `self-hpo-gpu` and use
`python -m baseline p-step --config configs/smoke_a100.yaml --output runs/p-new --execute`.
The cluster script also sets the pinned source paths, CUDA and local routing
variables; use it for the measured environment rather than substituting the
image's unrelated Slime checkout.

## H-step, outer loop and resume

Use the same submission with `BASELINE_PHASE=h-step`, and add
`PROPOSER_MODEL=CLAUDE_5puknv` (the already checked external enterprise Messages
route). H-step itself only needs one serving GPU; reduce allocation to one GPU,
8 CPUs and 96 GiB. It never updates the proposer or policy weights.

`BASELINE_PHASE=run` uses the configured two outer rounds and four-cell dev
comparison, requiring the 4-GPU training allocation. Set a deliberately bounded
`BASELINE_TIMEOUT` large enough for those real rollouts. This is a separate
experiment, not implicit continuation of the one-update smoke.

To resume an interrupted outer experiment, retain the same code/config/data and
set `BASELINE_PHASE=resume` plus `BASELINE_OUTPUT=<existing artifacts directory>`.
The loop preserves failed attempts and resumes completed stages/pairs; it does
not claim arbitrary mid-optimizer fault recovery. Native Megatron checkpoints
are retained separately from exported HF policies.

## Read actual results

Artifacts and live logs are under `runs/train-smoke-20261031/job-<ID>/`, with a
pointer in `/data/results/rsi-coevolve-<ID>.txt`. The P-step writes native Slime
`train.log`, TensorBoard scalars, token/logprob/mask provenance, trusted grading
results, dense HF export checks and a fresh-server reload rollout. A successful
P-step `result.json` requires finite nonzero gradient evidence, changed language
model tensor probes and export/reload validation. Scores may still decrease.

All-equal task rewards, infrastructure failures and unchanged parameters are
reported explicitly; they are not replaced with simulated training gains.
See [STATUS.md](STATUS.md) for measured acceptance, including failed attempts.
