# Current baseline verification — Tencent adaptation

**CPU contracts verified; real task execution and H/P learning remain NOT RUN.** The active task/grade transport is Tencent AGS/E2B. The current process has no E2B key/template/domain or image mappings, no visible GPU, and no configured external proposer. No sandbox, paid API call or GPU job was launched.

## Implemented and checked

- Single actual mini-SWE-agent for search, RL, dev and final generation; native Slime dense 9B GRPO remains unchanged.
- Tencent proxy/direct routing, class-level envd reconnect, bounded commands, cleanup, non-root task preparation, fresh graders, explicit image mappings and registry digest checks.
- All three original final verifiers accept an explicit Tencent environment factory. Tests exercise those original functions without rewriting their grading logic or using final-benchmark tasks as smoke data.
- Frozen training split remains 1,987 optimization / 249 dev / 248 audit. Only the raw file reference changed to a repository-relative path; task memberships and fingerprints did not change.
- Public checkout support: pinned dependency bootstrap, portable manifest, README, credential-free examples, ignored raw data/models/runs/local configuration. Existing old prototype and assets remain intact.

## Actual test results

- Baseline tests: **14 passed**. Eight original tests cover data/loop/checkpoint/hook/token contracts; six added cases cover Tencent routing, exit status, cleanup, image drift and the three original verifier callbacks. SDK/score fixtures are labeled test-only; they are not real Tencent rewards.
- Pinned Slime regression: **49 passed, 1 skipped**. The skipped case needs the unavailable SGLang tool parser. No upstream assertions were weakened.
- `pip check`, Ruff, bytecode compilation and Git diff checks passed. Latest log: `runs/tencent-adaptation/cpu-tests.log` (ignored, local evidence).
- Dependency bootstrap `--check --with-final-eval` passed against existing clean pins without cloning or upgrading them.
- An inventory of eight optimization task images and an actual blocked preflight were generated under `runs/tencent-adaptation/`. No images were transferred and no sandbox was created.
- Earlier CPU tokenizer rendering passed for the local 9B model (324 tokens, EOS 248046); it is not evidence of GPU inference.

Historical Orchard audit: 26 upstream tests passed and 10 failed because their mock import target used `orchestrator.sandbox_manager` rather than `orchard_env.orchestrator`. Those results remain in Git history and local audit records. Orchard is now an inactive reference, not a prerequisite for the Tencent baseline.

## Baseline Ready Checklist

| Requirement | Real status | Software evidence |
|---|---|---|
| Actual Tencent task and scoring | NOT RUN | Transport and original-grader callback fixtures pass |
| Real GRPO update | NOT RUN | Native dense launch/token/mask contracts pass |
| Save/export/reload/serve/rollout | NOT RUN | Dense export and source model/tokenizer checks pass |
| Actual proposer-driven Harness evolution | NOT RUN | Original gate/selection components pass CPU tests |
| Two real H/P rounds | NOT RUN | Interruption/resume and pair binding pass test-only two-round flow |
| Real four-cell cross-evaluation | NOT RUN | Same-task/seed and pair identity contracts pass |
| Verified / Multilingual / Pro V2 scores | NOT RUN | Separate adapters, pinned manifests and explicit Tencent callbacks |

Next required inputs are the approved Tencent template/domain/key, mirrored task images and host registry access, then a compatible GPU allocation for Qwen/Slime and proposer configuration for H-step. Start with one optimization-task oracle and one agent rollout before GRPO. See [Tencent setup](TENCENT_SANDBOX.md) and [RUNBOOK.md](RUNBOOK.md). No new research algorithms are required for these checks.
