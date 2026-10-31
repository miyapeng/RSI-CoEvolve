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
