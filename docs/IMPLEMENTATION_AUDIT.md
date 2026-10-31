# Implementation audit — 2026-10-31

Baseline is an independent **Self-HPO Git worktree**, branch `research/baseline-9b`, at `../self-hpo-baseline`. Existing `../harness-policy-coevolution` (937b5e8) stays intact; its Item/trigger/target-selection loop is not imported. All inspected repositories were clean. Manuscript and historical runs remain unchanged.

| Asset | Observed | Decision |
|---|---|---|
| Self-HPO | local MIT checkout 3c3d0bd93a8a5a72399c855783460b141ea27d49 | Main worktree, seed mini agent, experience/parent/fire/selection logic |
| Official Slime | local Apache-2.0 v0.4.0, 0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e | Only trainer: native dense 9B config, OpenAI trajectory adapter, GRPO, HF export |
| Original Self-HPO P-step | 35B MoE; custom facade runs actions BEFORE harness step | Do not run for 9B; use the same full mini agent for train and eval via native Slime adapter |
| Self-HPO-RL-Data | 2,484 rows, unique IDs; local raw SHA256 77638f74cf55b43b60af92eb042e40b13b350412836fad93b3c6bcee60e742e0 | No download; repository-grouped fixed split; retain source-specific grading |
| Qwen3.5-9B | models/Qwen3.5-9B -> shared BF16/F32 weights; four shards | No copied weights; config/tokenizer/index validation |
| Conda harness-policy | Python 3.12, torch 2.9.1+cpu, mini 2.2.6+selfhpo, swebench 5.0.1 | CPU implementation/tests, no venv |
| Conda v5-miles | torch 2.11.0+cu130, Ray 2.59.0, megatron-core 0.18.0; no SGLang | Not a verified Slime training environment; do not upgrade in place |
| Hardware | nvidia-smi: No devices found; no nvcc/Docker/kubectl | Actual RL and task execution NOT_RUN |
| Orchard | no endpoint/key/SDK found initially | Acquire only missing official SDK source; no server deployment/paid jobs |
| Tencent E2B | previous prototype has adapter; SDK installed, no env configuration | Retained intact in old project; baseline uses Orchard, no automatic backend fallback |
| Proposer | claude CLI 2.1.299 present, relevant API env keys absent | API JSON proposal mode; no host tool permissions or implicit billing |
| Agent Lightning / veRL | AGL local source, no verified GPU RL run | Not an active baseline backend |

The new baseline deliberately uses the already pinned official Slime v0.4.0 instead of Self-HPO's bundled 35B fork: native synchronous OpenAI capture permits the identical mini `run/step/execute_actions` path during training and evaluation, and the pin already contains dense 9B HF import/export. GPU compatibility is a pending execution check, not an asserted reproduction. Upstream optimizer/loss code is unchanged.

Scope: fixed H/P schedule; three data roles; code+prompt harness snapshots; source-specific real reward; four-cell dev diagnostics; durable pair checkpoints. No Item Registry, dynamic trigger, KL, pruning, target selection, population, custom loss, or automatic task submission.

## Completed local foundation checks

- Dedicated Conda `self-hpo-baseline` was cloned from existing CPU environment. Only this environment uses Self-HPO's original `swebench==4.1.0`; the old prototype still uses 5.0.1. Orchard SDK installed from local pinned source. Missing SDK dependencies installed; `pip check` passed.
- Frozen split: 1,987 optimization / 249 dev / 248 audit. Source-stratified repository components; zero cross-split task/repository/exact-patch/normalized-problem/image overlap. Semantic near-duplicates are not exhaustively ruled out.
- All 2,484 task specs construct using the original Self-HPO test builders. Parser distribution: pytest 1,895; Scale-SWE 584; pytest_v2 5. Construction is not evidence that any task image or test executed.
- Agent and grader use separate Orchard sandboxes. Public model projection excludes gold/test patch, test IDs and private grader metadata. The seed git-diff submission protocol is retained; agent source history is replaced with a clean baseline commit inside its fresh sandbox. This hardening is an explicit adaptation, still awaiting image-level validation.
- Eight CPU tests passed, including real mini + native Slime HTTP/token capture over scripted model/sandbox fixtures and auxiliary-only Dense export validation. Fixtures are not RL or benchmark results. Full final validation and known upstream test failures are in [STATUS.md](STATUS.md).

## Final asset decisions

- Only newly cloned repository: Orchard-Agentic (MIT), `bd8da32b3a075128700caab75bbc491fb6fdc16e`. Self-HPO, official Slime, Pro V2 verifier, three final datasets and all model files were already local. Reference repositories remain clean.
- The 9B source is `/user/lumingcong/models/Qwen3.5-9B`, referenced through ignored links; no full model was copied. Four safetensors shards / 775 tensor headers, index sizes and bounded probes were checked. Local tokenizer renders a multi-turn bash/tool-response prompt successfully (324 tokens; EOS 248046); this does not validate a GPU forward or SGLang parser.
- Data sources as actually labeled: 801 `nebius/SWE-rebench-V2`, 584 `AweAI-Team/Scale-SWE`, 1,099 rows missing a dataset label but carrying the older install-config structure. The missing labels remain explicitly marked in the manifest instead of inventing a confirmed origin. Original source-specific test builders are preserved.
- New CPU Conda env now has Transformers 5.12.1 / safetensors 0.8.0 (matching versions already present in the other inspected GPU environment); tokenizers resolved to 0.22.2 in this new env only. Mini editable import points into this worktree. Old prototype/Conda environment remain independent.
- Runtime candidate edits are deliberately limited to actual mini hooks. The official Slime adapter executes those same methods rather than copying Self-HPO's incompatible training facade. Model/config files and task grading are not proposer-editable.
- Real stages remain blocked by missing visible GPU, SGLang/Megatron/TE/Ray/TensorBoard in this environment, Orchard endpoint/key and external proposer configuration. No task submission, benchmark run, model download or paid API call was made. Do not interpret prepared commands as a validated GPU stack.
