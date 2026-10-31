# Actual verification — 2026-10-31

**One real Scale-SWE optimization task completed and scored successfully through
Tencent, using an explicitly selected API execution model. No 9B GPU inference,
GRPO update, harness search or two-round H/P experiment has run.**

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
not a currency cost estimate; monetary cost was not retrieved. No GPU job, bulk
mirror operation or final benchmark evaluation was launched.

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
  the baseline optimization pool. One task mirror was independently verified;
  the remaining mirrors are not thereby certified available.
- The dataset split remains 1,987 optimization / 249 dev / 248 audit. No final
  Verified, Multilingual or Pro V2 task was used for this smoke.
- Observed working public Tencent domain/template and precise commands are in
  [LIVE_SMOKE.md](LIVE_SMOKE.md). UID isolation, root-private files and lack of
  injected credentials were checked in the synthetic agent. Full provider egress
  isolation and running-image digest attestation remain unverified.

## Software checks

- Baseline CPU tests: **19 passed** after the native Messages and JSON-fence fixes.
- Pinned Slime regression: **49 passed, 1 skipped** (unavailable SGLang parser).
- `pip check`, Ruff, compilation, shell syntax and Git diff checks pass.
- Latest evidence: `runs/live-smoke-20261031/cpu-tests-release.log`.
- Historical Orchard/mock import failures remain recorded in history; Orchard is
  not an active runtime prerequisite. Upstream test assertions were not weakened.

## Baseline Ready Checklist

| Requirement | Status |
|---|---|
| Actual Tencent task image and original scoring | VERIFIED on one optimization task |
| Agent tool loop, patch and separate scoring | VERIFIED with API models; 9B NOT RUN |
| Strong external proposer API/protocol | VERIFIED; full H-step candidate search NOT RUN |
| Real Qwen3.5-9B GRPO update | NOT RUN — no visible GPU / compatible training stack |
| Checkpoint save/export/reload/serve | NOT RUN on a trained checkpoint |
| Actual harness evolution and P-step hook acceptance | NOT RUN; CPU compatibility tests pass |
| Two real H/P rounds / four-cell evaluation | NOT RUN |
| Verified / Multilingual / Pro V2 scores | NOT RUN; adapters remain independent of training |

Next: obtain a Slime-compatible GPU allocation, run 9B serving/tool smoke and one
real GRPO update, then H-step and the two-round baseline. Additional task image
mappings must be inspected/pinned as the chosen search/dev budget grows.
