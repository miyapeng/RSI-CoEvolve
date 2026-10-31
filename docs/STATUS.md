# Baseline verification — 2026-10-31

**CPU contracts verified; the real H/P baseline is not yet experimentally validated.** The current allocation exposes no GPUs and has no configured Orchard endpoint/key or external proposer configuration. No real task reward, GPU update, served checkpoint reload, API proposal or final benchmark result is reported.

## Implemented phases and actual evidence

| Phase | Input → output | Actual status |
|---|---|---|
| 0: audit | Existing repositories/assets → independent Self-HPO worktree and Conda environment | DONE. Old prototype and references preserved; only missing Orchard source cloned. |
| 1: data/environment/reward | Existing 2,484-row raw JSONL → frozen 1,987 / 249 / 248 split, source-specific test specs, Orchard execution/grading adapters | CPU validation passed for all 2,484 specs and split checks. Actual image launch, patch and test execution NOT_RUN. |
| 2: fixed-H P-step | Dense 9B source + seed + optimization data → native Slime launch plan, export/reload validation code | Plan prepared; no MoE arguments or final eval data. Actual GRPO/save/export/reload/rollout NOT_RUN. |
| 3: fixed-policy H-step | Training-side traces → original diagnosis, proposal, parent/fire gates and paired selection | CPU integration passed with explicit scripted model/sandbox/score fixtures. Actual API proposals/candidates NOT_RUN. |
| 4: two rounds / recovery | Accepted pair → two persisted rounds and correct next-round inputs | Two-round interruption/resume and policy/harness binding passed in test-only fixtures. Two real rounds NOT_RUN. |
| 5: diagnostics / switches | Same loop → Base/H-only/P-only/HPO plus paired dev records and report | CPU mode/cross-pair contracts passed. Real cross-evaluation NOT_RUN. |
| 6: final evaluation preparation | Existing local Verified/Multi/Pro V2 data → pinned independent plans and verifier imports | Offline hashes/counts and all three verifier imports passed. All benchmark execution NOT_RUN. |

## Tests actually run

- `bash scripts/baseline/test_cpu.sh` in Conda `self-hpo-baseline`: **8 passed (24.34 s)**. Tests cover fixed data roles, no gold in model requests, full mini hooks on search/train/eval, rejected candidate retention, upstream experience/gate/selection, fixed dense training plan, auxiliary-only export, score-parser fixtures, two-round interruption/resume, baseline switches, native Slime HTTP/token/logprob/mask alignment, and production rejection of injected simulated operations. Assertions were not weakened to pass.
- Same script runs unmodified pinned Slime `tests/test_agent/test_adapters.py` and `test_trajectory_manager_branching.py`: **49 passed, 1 skipped (1.29 s)**. The skip needs the unavailable SGLang tool parser. Passing scripted parser/capture tests does not validate Qwen on a real SGLang GPU server.
- `pip check`, Ruff and `compileall` passed. Complete latest output: `runs/validation/cpu-tests.log`.
- Actual local 9B tokenizer loaded without network or model tensors; multi-turn tool-call/tool-response rendering passed (324 tokens, EOS 248046). `runs/validation/tokenizer.json`. GPU forward/parser NOT_RUN.
- Orchard upstream timeout/network tests: **26 passed, 10 failed (1.10 s)**. The 10 failures are in `TestSandboxManagerNetwork`: their mock targets import old `orchestrator.sandbox_manager`, while the installed package is `orchard_env.orchestrator`, producing `ModuleNotFoundError: orchestrator`. These are still failed upstream tests. SDK source/tests were not edited; no blanket "all upstream tests pass" claim. Observed summary: `runs/validation/orchard-upstream-observed.json`.

Reproduce the Orchard check independently (it is expected to expose that upstream failure):

```bash
cd /user/miyapeng/rsi_evolve/references/Orchard-Agentic/orchard_env
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /user/miyapeng/miniconda3/envs/self-hpo-baseline/bin/python -m pytest \
  -p pytest_asyncio.plugin tests/test_exec_timeout.py tests/test_network_control.py -q
```

The old Agent Lightning/veRL prototype was not modified or substituted as a second training backend. No unexecuted tests from that project are claimed here.

## Prepared artifacts and cost accounting

- `runs/validation/preflight/preflight.json`: actual missing-resource report, model sparse fingerprint, installed package and repository versions.
- `runs/p-plan-dense/`: 16 selected optimization tasks, immutable harness, real native Slime command and checkpoint destination; **status NOT_RUN**.
- `runs/final-plans/{verified,multilingual,pro-v2}/plan.json`: 500 / 300 / 642 tasks, exact dataset commits and grading metadata; **status NOT_RUN**.
- `runs/validation/packages.json`: actual CPU package versions; `environment-baseline.yml`: direct dependency setup.
- Real task executions: **0**. GPU training steps: **0**. External proposer calls / evaluated live candidates: **0**. Allocated GPUs used: **none**. No paid job or API call was submitted. Scores, loss, real token cost, GPU runtime and monetary expenditure are not measured experiment results; they remain absent rather than synthetic zeros in metrics. CPU test times above are observed test durations, not training costs.

## Baseline Ready Checklist

| Requirement | Real verification | Available evidence |
|---|---|---|
| Real task scoring, empty/failed/successful patches | **NOT_RUN** | Original grading builders/parsers + CPU fixtures; no live sandbox |
| Real GRPO parameter update | **NOT_RUN** | Native dense command and token/mask capture contracts |
| Save → export → reload → serve → rollout | **NOT_RUN** | Auxiliary export unit test and model/tokenizer structure checks; live chain required |
| Harness evolution with actual proposer/gate/reward | **NOT_RUN** | Original components exercised under scripted transports |
| Two actual H/P cycles with updated policy | **NOT_RUN** | Two-cycle state/recovery test only |
| Actual four-cell cross-evaluation | **NOT_RUN** | Pair binding and same-task/seed implementation verified in CPU tests |

Necessary next engineering work is bounded: obtain a compatible GPU allocation/image and Orchard endpoint/image access; run the optimization-only oracle/task smoke; resolve any image permission/parser/parallelism compatibility revealed there; then execute one GRPO export/reload and the two-round live run. The small Orchard upstream test import-path issue also remains recorded. No new research algorithms are prerequisites for these steps. Exact commands: [RUNBOOK.md](RUNBOOK.md).
