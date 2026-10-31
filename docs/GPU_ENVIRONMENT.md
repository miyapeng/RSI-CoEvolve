# GPU environment and image validation

The CPU development environment is already configured in Conda. GPU execution
needs a compatible CUDA stack **inside the scheduled Cybertron job**. Activating
the CPU environment inside a GPU container would select CPU PyTorch again.

| Asset | Purpose | Location / handling |
|---|---|---|
| Conda `self-hpo-baseline` | CPU development, dataset checks, API/Tencent smoke | Existing personal environment; unchanged |
| GPU job image | CUDA, PyTorch, SGLang, Megatron, Slime dependencies | Cybertron pulls a registered image onto the allocated node |
| Qwen3.5-9B weights | Policy parameters and tokenizer | Existing read-only model symlink; no additional model download |
| Tencent task image | Repository, build dependencies and tests for each SWE task | Runs in E2B; distinct from the GPU training image |

You do not need to download a training image onto the CPU development machine
before submitting a job. A registered image must be synchronized to the target
cluster; the scheduled node then pulls it from that cluster's registry. Image
availability and actual CUDA/model compatibility are separate checks.

## Observed resources, 2026-10-31

- Both authorized pools, `rl-train` and `ocr-train`, belong to `langfang_train`
  and advertise A100 resources. Availability changes; query again before launch.
- `ptdbench-slime-v0.3.2-cu129` (image ID `25607131`) is registered and its
  Langfang synchronization status was `Succeeded`.
- `ptdbench-slime-rc0` and `miles-minicpm5-rl-1030` are also synchronized there.
  They are candidates for inspection, not verified interchangeable environments.
- The baseline remains on Slime commit
  `0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e` (v0.4.0). The image's `v0.3.2`
  label is not evidence that it matches this source version.

The pinned upstream Dockerfile specifies SGLang `0.5.15.post1-cu129`, Megatron
`4b4acac9a1d28ea6829c8d4f566d75698a21249d`, Transformer Engine `2.18.0`, and
NCCL `2.30.7`, including upstream patches. This is the reference dependency
recipe, not a claim that a tested enterprise image has those versions. A100 is
SM80: the Qwen Gated DeltaNet path must use FLA, not FlashQLA's SM90-only path.
The old Self-HPO B200 `trtllm_mha` serving defaults are not used.

## Small image probe

`scripts/baseline/probe_gpu.py` runs independent subprocess checks for CUDA
backpropagation, Transformer Engine, FlashAttention, Qwen's delta-rule kernel,
SGLang RMSNorm, local tokenizer/weight-file access, and imports through the pinned
Slime Qwen implementation. Each check has a timeout and its own log. A failed
check remains a failure; the process returns nonzero. Missing package versions
are null. This does not train or claim a GRPO update.

`scripts/baseline/run_gpu_probe.sh` caps runtime at 900 seconds, writes to
`/data/results`, and archives small reports under the owner's ignored `runs/`
directory. It does not install dependencies, download weights or call APIs.

Submission uses the existing authenticated cctl profile. `BILLING_ACCOUNT_ID`
below is the account authorized for the experiment; it is not an API key.

```bash
cd /user/miyapeng/rsi_evolve/self-hpo-baseline
cctl billing-account list
cctl image get 25607131
cctl node list langfang_train --pool ocr-train --only-available

# Preview first; remove --dry-run to submit this small allocation.
cctl job create --dry-run \
  --project myp-test-2 --cluster langfang_train --resource-pool ocr-train \
  --billing-account-id "$BILLING_ACCOUNT_ID" \
  --image ptdbench-slime-v0.3.2-cu129 \
  --priority PREEMPTABLE --gpu 1 --gpu-model a100 --cpu 4 --memory 32 \
  --entry 'bash /user/miyapeng/rsi_evolve/self-hpo-baseline/scripts/baseline/run_gpu_probe.sh' \
  --description 'RSI-CoEvolve bounded GPU compatibility probe, no RL update'

cctl job get JOB_ID
cctl diagnosis get-live JOB_ID
cctl job logs JOB_ID --page-size 100
```

On this development machine, inherited HTTP/HTTPS proxies interfere with the
Cybertron management API. The actual submission removed proxy variables **only
for the cctl subprocess**. This does not change Tencent sandbox networking.
The installed CLI accepts memory in integer GiB and image pagination via
`--page-token`. Its `job logs` command does not accept `-o json`.

## Actual 9B serving result and repeat command

Job **1427589** succeeded on one A100-SXM4-80GB, driver 535.129.03, with the
existing image and existing model files. The GPU loaded the real 9B weights and
returned the requested `bash` tool call using `qwen3_coder` / `qwen3` parsers.
Reported usage: 297 prompt and 61 completion tokens. Startup, request and
cleanup took 97.754 seconds. No task patch, score, gradient or checkpoint update
was produced by this test. SGLang's shutdown logs contain SIGTERM/SIGQUIT during
our owned-process cleanup; the request and job completed successfully.

The measured configuration was derived from the single baseline YAML, saved in
ignored `configs/gpu_smoke.local.yaml` and archived with the run. To reproduce:

```bash
# Run this small config preparation in the existing development Conda env.
python - <<'PY'
from pathlib import Path
import yaml
c = yaml.safe_load(Path('configs/baseline_9b.yaml').read_text())
c['serving'].update(tensor_parallel=1, port=18409,
                    startup_timeout=600, mem_fraction_static=0.65)
c['runtime']['max_context_tokens'] = 4096
Path('configs/gpu_smoke.local.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
PY

cctl job create --dry-run \
  --project myp-test-2 --cluster langfang_train --resource-pool ocr-train \
  --billing-account-id "$BILLING_ACCOUNT_ID" \
  --image ptdbench-slime-v0.3.2-cu129 \
  --priority PREEMPTABLE --gpu 1 --gpu-model a100 --cpu 4 --memory 64 \
  --env GPU_SERVE_CONFIG=/user/miyapeng/rsi_evolve/self-hpo-baseline/configs/gpu_smoke.local.yaml \
  --entry 'bash /user/miyapeng/rsi_evolve/self-hpo-baseline/scripts/baseline/run_gpu_serve.sh'
```

Remove `--dry-run` to submit. On an already allocated compatible GPU, the same
entry is `python -m baseline.serving --config configs/gpu_smoke.local.yaml
--output /data/results/9b-serving-smoke`. The script sets the local mini agent's
source on `PYTHONPATH`; it does not require Tencent credentials or invoke any
task sandbox. Full task execution still uses `python -m baseline rollout`.

## Full Megatron source, distinct from megatron-core metadata

The first operator probe, job **1427569**, passed all GPU kernels but failed
`import megatron.training`. The image has full source at `/root/Megatron-LM`,
but it is not in the default Python path and has a different commit from the
baseline reference recipe. A package version named `megatron-core` is not proof
that the `megatron.training` package is importable.

The required source now exists at adjacent `../references/Megatron-LM`, commit
`4b4acac9a1d28ea6829c8d4f566d75698a21249d`, with only the two original Slime
patches applied. Patch hashes and license are in [UPSTREAM.md](UPSTREAM.md).
Do not overwrite the neighbouring MiniCPM project's Megatron or Conda packages.

For a **fresh missing checkout only**, the actual preparation was:

```bash
git init ../references/Megatron-LM
git -C ../references/Megatron-LM remote add origin git@github.com:NVIDIA/Megatron-LM.git
git -C ../references/Megatron-LM fetch --depth 1 origin 4b4acac9a1d28ea6829c8d4f566d75698a21249d
git -C ../references/Megatron-LM checkout --detach FETCH_HEAD
git -C ../references/Megatron-LM apply --check ../slime/docker/patch/v0.5.15.post1/megatron.patch
git -C ../references/Megatron-LM apply ../slime/docker/patch/v0.5.15.post1/megatron.patch
git -C ../references/Megatron-LM apply --check ../slime/docker/patch/v0.5.15.post1/megatron-sglang-aligned.patch
git -C ../references/Megatron-LM apply ../slime/docker/patch/v0.5.15.post1/megatron-sglang-aligned.patch
```

For the operator probe, pass the job environment variable
`MEGATRON_ROOT=/user/miyapeng/rsi_evolve/references/Megatron-LM` to prepend this
source. For a future training job, explicitly prepend that directory to
`PYTHONPATH` before starting the baseline CLI; `train_worker.py` preserves and
forwards this path to Ray workers. Do not infer full GRPO compatibility merely
from this source preparation or an import check.

The recheck, job **1427598**, **succeeded**: all eight checks passed in 138.826
seconds, including the pinned Slime training/model imports. The image's package
metadata still reports `megatron-core==0.16.0rc0`, while the explicit source path
selects the pinned full checkout; record both rather than confusing metadata with
the imported code. Import logs retain TorchAO/ModelOpt compatibility warnings.
No quantized checkpoint loading, distributed GRPO or optimizer state restore has
been tested. Three sequential one-GPU probes consumed approximately 355.49 seconds
of measured script runtime in total, excluding image pulling and scheduling;
monetary cost was not retrieved. All three jobs are finished.

## Conda and subsequent training

Do not modify the existing `v5-miles` environment or update all packages to latest.
The neighbouring project's successful individual GPU operator checks are useful
prior evidence, but do not establish that its full Slime/Qwen chain works.
The first image probe deliberately uses the image's Python to establish what
the image actually supplies. This is an audit, not a switch to a venv workflow.

A separate `self-hpo-gpu` Conda environment now preserves that tested
Torch/CUDA/operator stack and installs the baseline's pinned agent/scoring
dependencies and local Slime/Megatron sources. See [GPU_CONDA.md](GPU_CONDA.md)
for the actual preparation, package discrepancies and tested cluster routing. `environment-baseline.yml` is the **CPU** recipe;
do not apply its CPU Torch install command to a GPU environment. If rebuilding
an image is necessary, pin Slime and Megatron source, retain the matching
upstream patches, record the built image digest and rerun these checks.

GPU operators, real 9B serving/tool calls, an actual GRPO update, and exported
checkpoint reload are distinct acceptance levels. See [STATUS.md](STATUS.md)
for the measured level reached; none can be inferred from a successful image
pull or from `torch.cuda.is_available()` alone.
