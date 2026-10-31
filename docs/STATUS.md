# Actual verification — 2026-10-31

**One real Scale-SWE optimization task completed and scored through Tencent with
an API execution model. Separately, Qwen3.5-9B loaded on A100 80GB and passed a
real tool-call request. Native Slime then collected four real 9B/Tencent
trajectories and scores, but that attempt failed before the optimizer. A
verified GRPO update, harness search and two-round H/P execution are still pending.**

## Live results

| Run | Actual result |
|---|---|
| Synthetic coding fixture | Same seed mini agent repaired a function; separate real grader passed 6 private checks. Qwen3.8-Max API, 6 tool calls, 12,327 reported tokens, 44.66 seconds. Generic unpinned image; not a benchmark task. |
| `open-feature_python-sdk_pr329` grader oracle | Existing Tencent task image, registry digest pinned. Empty patch: 0 without test execution; invalid patch: 0 on failed application; reference patch: 1 with the original source-specific tests executed. 13.85 seconds. No reference patch sent to any policy. |
| First real-task API rollout | Qwen3.8-Max timed out during a model request under the 300-second agent budget. `APITimeoutError`, null score; failure and trajectory retained. |
| Next invocation | Native Opus response enclosed its JSON in a code fence. Strict bare-JSON parsing failed before any task sandbox was created. Fixed with explicit single-fence unwrapping plus a regression test; no arbitrary prose extraction. |
| Corrected real-task API rollout | GPT-6.1-Terra gateway model, same seed agent and original source grader: **score 1.0**, 7 tool calls, 31,518 reported tokens, 40.35 seconds including API check and grading. One FAIL_TO_PASS and 42 PASS_TO_PASS tests succeeded. |

The API gateway names/IDs are recorded, not claimed as independently attested
provider checkpoints. The frozen external proposer route is gateway Opus 5.5
(`CLAUDE_5puknv`), native Anthropic Messages. The real-task API execution model is
`GPT_15kraa`; it is **not** the baseline's trainable Qwen3.5-9B. Short API checks
validate transport/formatting, not proposer-driven harness evolution.

Results are single-task engineering checks, not estimates of model success rate
or RL improvement. Neither failed attempt was discarded. Reported token usage is
not a currency cost estimate; monetary cost was not retrieved. No bulk mirror
operation or final benchmark evaluation was launched. GPU checks are recorded
separately below.

Evidence: [compact measured record](validation/live_smoke_20261031.json), local
`runs/live-smoke-20261031/` (raw requests/responses, tool events, patches, grading
logs and failures; ignored by Git). Nine recorded sandbox IDs were independently
queried and all returned 404 after cleanup. The timed-out task's context cleanup
completed but its error record lacked the sandbox ID; error provenance now also
captures that ID. Keys and registry authentication were held only in process
memory; they are absent from these files and the repository.

## Reused assets and remaining boundary

- Existing Conda environment, frozen Self-HPO data, source graders, seed harness,
  E2B client and neighbouring R2E mirror naming convention reused. No new trainer,
  agent loop, model download or container image transfer.
- The neighbouring prepared Scale-SWE file has 2,263 rows; 103 IDs/images overlap
  the baseline optimization pool. Three selected task mirrors have verified registry digests;
  only the open-feature task has passed a source-grader oracle so far.
- The dataset split remains 1,987 optimization / 249 dev / 248 audit. No final
  Verified, Multilingual or Pro V2 task was used for this smoke.
- Observed working public Tencent domain/template and precise commands are in
  [LIVE_SMOKE.md](LIVE_SMOKE.md). UID isolation, root-private files and lack of
  injected credentials were checked in the synthetic agent. Full provider egress
  isolation and running-image digest attestation remain unverified.

## GPU checks

The enterprise image `ptdbench-slime-v0.3.2-cu129` was already synchronized to
Langfang. The jobs used the authorized `ocr-train` pool, one A100 80GB at a time,
and PREEMPTABLE priority. Existing model weights were read without copying.

- Job **1427569**: real Torch, Transformer Engine, FlashAttention and FLA/Qwen
  delta-rule backpropagation passed; SGLang RMSNorm matched its reference;
  all four local 9B shards/tokenizer were readable. The job correctly **failed**
  its aggregate check because `megatron.training` was absent from the image's
  default Python import path. Measured probe runtime: 118.910 seconds.
- Job **1427589**: **Succeeded**. Real SGLang 0.5.15.post1 loaded Qwen3.5-9B on
  A100 80GB with TP=1, context=4096, memory fraction=0.65. The shared server
  returned `bash({"command":"printf baseline_tool_smoke"})`, finish reason
  `tool_calls`, 297 prompt + 61 completion tokens. Runtime including startup
  and cleanup: 97.754 seconds. The command was parsed, not executed on the host.
- Observed GPU driver 535.129.03; image Python 3.12.3, Torch 2.11.0+cu129,
  Transformer Engine 2.16.1, FLA 0.4.2, NCCL 2.28.9. These are measured versions,
  not the complete pinned Slime reference stack.
- Job **1427598**: **Succeeded**, all eight checks passed after explicitly adding
  the pinned full Megatron source and original Slime patches to the Python path.
  This includes imports of the fixed Slime Megatron arguments and Qwen3.5 model
  implementation. Runtime: 138.826 seconds. Existing operator versions were
  retained; no image rebuild or package upgrade was needed for these checks.

Details and reproduction: [GPU_ENVIRONMENT.md](GPU_ENVIRONMENT.md). Raw logs,
job requests and reports live under ignored `runs/gpu-smoke-20261031/`.
Public compact evidence: [gpu_smoke_20261031.json](validation/gpu_smoke_20261031.json).
The initial probes used image Python. Subsequent work prepared the isolated
`self-hpo-gpu` Conda prefix; see [GPU_CONDA.md](GPU_CONDA.md).

## Native training integration

- Job **1427683**: cluster-side Tencent grader oracle passed (empty/invalid/reference
  scores 0/0/1, 43 tests for the reference). Both sandboxes were cleaned up.
- Job **1427689**: real 9B/Tencent execution reached 8 tool calls, then exceeded
  the initial 8K context; no score or update claimed.
- Job **1427719**: native SGLang launched; the Qwen Processor rejected string
  dataset prompts. The adapter now emits conversation-form records and was
  checked with the actual local Processor and native Slime Dataset.
- Job **1427743**: missing declared `straw-queue` dependency blocked native training.
  Pinned dependency installation and four native training imports now pass.
- Job **1427762**: four NCCL ranks, HF actor/reference loading, weight sync and
  **four real fresh trajectories** ran. All hit the 20-step limit and scored 0
  (training reward -1). Batch conversion then failed on unavailable top-p support
  IDs, before an optimizer update. The adapter now requires top_p=1 consistently
  and applies unchanged native GRPO normalization to complete trajectories.
  The retained 80 segments / 17,253 trainable tokens pass native conversion.

Raw evidence is under `runs/train-smoke-20261031/`; compact public records:
[cluster oracle](validation/cluster_oracle_20261031.json),
[first native trajectories](validation/first_native_rollouts_20261031.json).
No old negative trajectories are silently reused as fresh on-policy samples.

## Software checks

- CPU baseline: **22 passed, 1 skipped**. The extra native BatchBuilder test
  requires Ray and runs separately in the GPU Conda prefix without a GPU.
- GPU Conda native adapter/group normalization: **2 passed**.
- Unmodified Slime agent regressions: **49 passed, 1 skipped**.
- Unmodified Slime DP scheduling / HF checkpoint saver tests: **15 passed**.
- CPU `pip check`, Ruff, compilation, shell syntax and Git diff checks pass.
  GPU image package metadata discrepancies remain documented in GPU_CONDA.md;
  its `pip check` is not claimed clean.

## Baseline Ready Checklist

| Requirement | Status |
|---|---|
| Actual Tencent task image and original scoring | VERIFIED on one optimization task |
| Agent tool loop, patch and separate scoring | VERIFIED with API models; 4 real 9B trajectories scored 0 at step limit |
| Qwen3.5-9B GPU load and tool-call parser | VERIFIED on A100 80GB, job 1427589 |
| GPU kernels and pinned Slime/Megatron imports | VERIFIED, job 1427598; full training remains unverified |
| Strong external proposer API/protocol | VERIFIED; full H-step candidate search NOT RUN |
| Real Qwen3.5-9B GRPO update | ATTEMPTED; failed before optimizer, fixes under validation |
| Checkpoint save/export/reload/serve | NOT RUN on a trained checkpoint |
| Actual harness evolution and P-step hook acceptance | NOT RUN; CPU compatibility tests pass |
| Two real H/P rounds / four-cell evaluation | NOT RUN |
| Verified / Multilingual / Pro V2 scores | NOT RUN; adapters remain independent of training |

Next: finish the Slime training environment and run one real GRPO update,
then H-step and the two-round baseline. Additional task image
mappings must be inspected/pinned as the chosen search/dev budget grows.
