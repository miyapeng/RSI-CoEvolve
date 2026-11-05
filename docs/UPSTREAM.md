# Upstream provenance

| Repository | Pinned commit | License | Use / modification |
|---|---|---|---|
| https://github.com/MSR-Orchard/Self-HPO | 3c3d0bd93a8a5a72399c855783460b141ea27d49 | MIT (LICENSE); mini MIT (mini-swe-agent/LICENSE.md); bundled Slime Apache-2.0 | Main worktree. Seed `harnesses/seed`, mini agent/actions/submission semantics retained. H-step experience/parent/fire/record scripts reused; path/config integration changes recorded in Git. Source-specific P-step grader functions extracted from RL imports, semantics retained. Original 35B shell launcher is historical and not active. |
| https://github.com/THUDM/slime | 0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e (v0.4.0) | Apache-2.0 | Existing adjacent dependency, unchanged: scripts/models/qwen3.5-9B.sh, slime/agent/adapters, trajectory capture, Megatron GRPO, HF checkpoint saver. |
| https://github.com/NVIDIA/Megatron-LM | 4b4acac9a1d28ea6829c8d4f566d75698a21249d | BSD-3-Clause plus per-file third-party notices in LICENSE | Newly fetched exact source pin from Slime's Docker recipe, into adjacent `references/Megatron-LM`; applied only the two pinned Slime upstream patches described below. |
| https://github.com/microsoft/agent-lightning | d381995396274039f2bb1cbe5ff42ac8067f4e47 | MIT | Inspected existing asset only; not imported or maintained as another trainer. |
| Local harness-policy-coevolution | 937b5e8 | MIT, upstream notices retained there | Reuse dataset/model links and lessons from native Slime adapter integration. Research algorithms not imported. |

| https://github.com/microsoft/Orchard-Agentic | bd8da32b3a075128700caab75bbc491fb6fdc16e | MIT | Acquired in the initial audit. Preserved as an unchanged reference; not required by the active Tencent baseline. |
| https://github.com/scaleapi/SWE-bench_Pro-os | 66f92766bba642462d4bbe5479e83f91f9211862 | MIT, Copyright 2026 Scale AI | Existing adjacent dependency; Pro V2 per-task tests used by the original mini verifier. No tests copied or modified. |

Default branches observed: Self-HPO, Slime, Agent Lightning, Orchard-Agentic and SWE-bench Pro use `main` (Slime checked with `git ls-remote --symref`, remote HEAD then `540ea919e45b8b929198abce008a103be9cfc779`). Slime remains pinned to the existing local v0.4.0 checkout, not that newer HEAD. Megatron's later audit is recorded below. This worktree shares Self-HPO Git history; dependency repositories remain outside this worktree's commits. Original root MIT, mini MIT and bundled Slime Apache-2.0 licenses are preserved. See [third-party notices](../THIRD_PARTY_NOTICES.md).

## Actual file-level reuse and adaptations

| Upstream path | Local use/change |
|---|---|
| `harnesses/seed/{agent.py,config.yaml}` | Unchanged starting harness and prompt. Seed aliases `ProgressTrackingAgent`; no additional loop. |
| `mini-swe-agent/src/minisweagent/{agents/default.py,run/benchmarks/swebench.py,models/utils/actions_toolcall.py,environments/docker.py}` | Unchanged dependency calls: `run`, `step`, `execute_actions`, marks, formatting and submission detection. `baseline/runtime.py` supplies model/environment protocols, not another agent loop. |
| `scripts/h_step/propose_prompt.tmpl` | Original file unchanged. `baseline/search.py` renders it and appends the explicit API/code-edit contract and bounded optimization-only evidence. Original ADD-on-parent restriction retained for this baseline. |
| `scripts/h_step/build_experience.py`, `parent_check.py` | Only configurable working root (`BASELINE_WORKDIR`); retain original diagnosis and parent-retention check. |
| `scripts/h_step/fire_gate.py` | Configurable root; accepts collected real gate trajectories; discovers candidate class name. After job 1428745, distinguishes prompt text from actual host `open`/`Path` calls. Host access cannot bypass rejection by mentioning `self.env`. Trigger/submission checks retained; action markers are checked by `record_result.py`. |
| `scripts/h_step/record_result.py` | Configurable root and promotion margin; original paired scoring/frontier logic retained. |
| `scripts/h_step/{search_eval.py,run_h_step.sh,build_search_set.py}` | Audited and retained, not launched: original orchestration assumes original model/server/layout. Thin baseline caller invokes retained diagnostic/gate/selection components with its frozen splits. |
| `slime/examples/mini_swe/swe_reward.py` | Extracted original `make_test_spec_scaleswe` / `make_test_spec_swerebench_v2` to `baseline/grading_spec.py` so CPU/eval do not import GPU training. Original file imports those functions. |
| `slime/examples/mini_swe/swe_grading.py` | Original implementation preserved exactly in `baseline/grading_report.py`; old path re-exports it. `swebench==4.1.0` retained, no new reward definition. |
| `scripts/p_step/prepare_rl_data.py` | Replaced remote/default-data preparation with local frozen optimization/dev manifest conversion, never final-test data. |
| `scripts/p_step/run_p_step.sh` | Historical 35B command retained but removed default Verified training-time evaluation; points users to the new 9B entrypoint. Not used by this baseline. |
| `scripts/p_step/{prepare_ckpt.sh,export_policy.sh}` | Audited; MoE fusion path not used for Dense 9B. |
| Official Slime `scripts/models/qwen3.5-9B.sh`, `slime/agent/{adapters,trajectory.py}`, `train.py`, `backends/megatron_utils/{model.py,hf_checkpoint_saver.py}` | Reused unchanged through the adjacent pinned dependency. Native token/logprob/mask capture, optimizer, GRPO, HF import/export. Auxiliary-only HF shard completion is local, with no expert fusion. |
| Official Slime `tests/test_agent/_fakes.py` and tests | Existing test-only HTTP/token fixtures and original regression tests; no simulated results published as experiments. |
| Old local prototype `scripts/test_slime_upstream.sh` | Small environment-isolation recipe reused in `scripts/baseline/test_cpu.sh`; no upstream assertions or source altered. |
| `scripts/eval/{gen_traj.sh,score_traj.sh,serve_model.sh}` | Audited; default B200 backend and 35B orchestration not invoked. `baseline/final_eval.py` calls original mini verifier functions; generation always uses the shared runtime. |
| mini `swebench_verify_orchard.py`, `swebench_multilingual_verify_orchard.py`, `swebench_pro_verify_orchard.py` | Optional explicit `environment_factory` injection added for Tencent. Test construction, patch application, parsers and scoring retained. Pro V2 uses the official tests at the pin above. |

No HarnessAdapt-RL proprietary source or prompts copied. No Meta-Harness/RRSI/AHE engine merged. No Agent Lightning/veRL training code imported. Official URLs, local licenses and pins were checked; no global dependency upgrade performed. The existing old prototype contributed local assets and prior adapter lessons, not its Item/trigger/pruning algorithms.

## Tencent adaptation and publication

- `baseline/tencent.py` adapts the earlier local MIT prototype's `harness_policy/tencent_sandbox.py` at `937b5e8`: explicit image/digest checks, proxy/direct reconnect, bounded commands, exit status and cleanup. The old project remains unchanged. No private dual-harness source, credentials or configuration files were copied; the supplied integration protocol informed the implementation.
- [E2B SDK](https://github.com/e2b-dev/E2B): installed distribution **2.62.0**, Apache-2.0, used as a dependency. Installed source/signatures and the [official network API](https://docs.e2b.dev/network/internet-access) were checked. No Git commit is invented for a package release. Tencent provider enforcement remains unverified.
- `sandbox.py` retains common task preparation; `train_worker.py` forwards required Tencent environment variables to native Ray workers. No model, GRPO loss or agent-loop changes accompany this transport replacement.
- The frozen manifest's raw path is repository-relative; memberships, IDs, content hashes and seed are unchanged. Bootstrap fetches only missing pinned Slime/Pro source. Original Self-HPO README is preserved under `docs/upstream/` and in Git history; the repository README describes the 9B adaptation.

## Local asset discovery, 2026-10-31

Read-only reference: `/user/miyapeng/minicpm-v5-agentic-rl/repos/handoff-20261029/R2E-Gym-e2b/src/r2egym/agenthub/runtime/swe_rebench_v2.py`,
SHA256 `315b965d5981634638fb00974f1cc4b6e23ec20c6617766b462e97dbb3512953`.
The enclosing handoff directory is not a Git checkout; no commit is invented.
Its `normalize_tencent_image` convention guided discovery of existing mirrors.
No private runtime file or prompt was copied. Also inspected the neighbouring
prepared Scale-SWE metadata and prior image-presence reports. Reused only the
existing remote task image, after sandbox repository/base-commit verification
and authenticated registry digest inspection. The baseline keeps its own frozen
Self-HPO dataset split and original source grader. No final benchmark task was
used for this validation. See [live evidence](LIVE_SMOKE.md).

`baseline/proposer.py` is an original, thin HTTP protocol adaptation, not copied
provider code. Native Messages is needed by the observed enterprise Opus route;
it preserves the existing Self-HPO prompt renderer and candidate validation.

## GPU image audit, 2026-10-31

Read-only inspection of the neighbouring MiniCPM project's GPU operator report
and launcher informed the choice to test existing A100 images before building
another one. No private launcher, model implementation or loss was copied.
`scripts/baseline/probe_gpu.py` is an original bounded diagnostic using public
Torch/TE/FA/FLA APIs. `baseline/serving.py` moves the existing shared server code
out of `execution.py`, so the actual server can be checked without installing
task-grader dependencies. The execution entry re-exports the same functions;
there is still one agent implementation.

The already registered image `ptdbench-slime-v0.3.2-cu129` contains Megatron
source `1dcf0dafa884ad52ffb243625717a3471643e087` under `/root/Megatron-LM` and
Slime source `a3f500977f33b82de6ac0414c0c43abfe656c3e0` under `/root/slime`.
These **image observations** do not replace the baseline's explicit source pins.
Its default Python imported a `megatron-core` installation without the training
package. We fetched the missing reference-version source using SSH, without
changing the existing image, user environments or neighbouring repositories.

The fresh Megatron checkout is intentionally modified only by the unaltered
patches in pinned Slime `docker/patch/v0.5.15.post1/`, in this order:

| Patch | SHA256 |
|---|---|
| `megatron.patch` | `45eecebe0bcd5a3158aa6573983a7b700cce0d061cdf688b8fe6cc66b3ef7713` |
| `megatron-sglang-aligned.patch` | `669c582584f1b70c1da8eb1009a5ee9f2901a4557da67229c9d6a13189a6f57a` |

Both patches passed `git apply --check` before application. No custom optimizer,
loss or Megatron fix was added. This dependency remains outside the public Git
worktree; its full upstream LICENSE and file notices are retained. This source
preparation alone does not certify image compatibility or a successful update.
The official Megatron default branch was checked as `main` (remote HEAD then
`831aa66d42f0a259dabbf8b6be9cf89ad929b360`); the checkout remains at the
Slime-required pin above, not the remote HEAD.

The dedicated `self-hpo-gpu` Conda prefix imports the image's SGLang Python
package from source commit `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`, with
its upstream license retained. This is an environment dependency, not another
vendored agent/runtime. Image package provenance is recorded by
`scripts/baseline/import_image_environment.py`; local Slime and Megatron editables
override the image's versions. No upstream training loss was changed.

Slime's declared `straw-queue` dependency is installed as package **0.1.2**,
MIT, with repository metadata https://github.com/zhuzilin/straw. It is used
unchanged as an environment dependency; no source was copied into this project
and no Git commit is inferred from the wheel version.

## Agent trajectory training adaptation

Pinned Slime `slime/data/batch_builder.py` requires nucleus support token IDs
when `rollout_top_p < 1`, but its native OpenAI Agent Adapter captures sampled
token logprobs only. The baseline therefore uses `top_p: 1.0` consistently for
H/P/eval and rejects incompatible P-step configurations before launch. No native
assertion or loss was removed. This differs from Self-HPO's original 0.95 setting.

`baseline.slime_rollout.post_process_rewards` delegates to the unchanged native
`BatchBuilder._post_process_rewards` once per task group with one representative
per complete trajectory, then broadcasts to its chat-template segments. This
preserves GRPO trajectory group semantics when trajectories have different
segment counts; it is a data adapter, not a new loss. Native loss aggregation
already uses `rollout_id`. Tests compare against native normalization and verify
that an all-failure task never borrows advantage from another task.

The API proposer adaptation retains `scripts/h_step/propose_prompt.tmpl` and
adds the actual local hook validator plus mini agent step/submission signatures
as runtime contract evidence. Frozen sampling values are rendered from config.
Invalid JSON/schema or forbidden code is a recorded rejection, not an implicit
retry or a weakened gate. Gateway/transport failures remain fatal and explicit.
The old prompt's `Submitted(string)` statement is corrected during API prompt
rendering to the exit-message dictionary required by `DefaultAgent.run`.
The upstream prompt file itself remains retained. The producer and consumer
source snippets are supplied together to avoid guessing this protocol.
`fire_gate.py` retains its original minimum of two submitted trajectories.
The baseline's `export_gate_runs` gathers every gate repetition into its expected
trace layout. A one-task smoke explicitly uses three seeds; standard searches
use three tasks. The original gate assertion was not weakened for the smoke.

Pinned Slime `slime/agent/adapters/common.py::call_sglang_generate` caps output
to remaining context and returns an empty length-limited turn at exhaustion.
`baseline.runtime.Model` aligns direct SGLang evaluation with that behavior:
on its exact context-limit 400 diagnostic it uses the reported input count to
cap one request, without truncating visible history; exhaustion ends the task.
All other API failures propagate. The native adapter itself remains unmodified.

Job 1428745 exposed a false positive in the original fire-gate heuristic:
`patch.txt` in a model hint was treated as host file access. The post-run change
inspects actual `open` / `Path` calls instead, and removes the unsafe exemption
for code that also mentions `self.env`. The baseline's stricter AST edit boundary
still runs first. CPU regressions cover remote hints, direct host reads, Path
reads, and mixed remote/host calls. The at-least-two-submissions assertion and
all live candidate records are preserved. This fix is not an assertion of
general Python sandbox security or a retroactive candidate promotion.

## Official visual / GUI / interactive-game evaluation adapters (2026-11-04)

Vision2Web's optional Tencent adaptation adds an original process-local Docker
command bridge in `baseline/benchmarks/vision2web_e2b.py`; the official source
remains clean at the commit below. The directly invoked command surface was
audited in `core/sandbox.py`, `inference/adapters/claude_code.py`, and
`evaluation/{engine,functional_tester}.py`. The build helper invokes the
unmodified upstream `docker/Dockerfile` externally (not vendored). Runtime
dependency `e2b==2.62.0` retains the existing SDK dependency boundary; its installed
package metadata declares `License-Expression: MIT`.
The user's `Visual-Self-Verification/evaluate/vision2web/clusterx_transport/docker`
was read as a transport-design reference; no sibling source or prompts were
copied. The new bridge manages real Tencent lifetimes rather than reusing a
ClusterX job's local filesystem. Actual tests and remaining image blocker are
in `docs/VISION2WEB_TENCENT.md`.

All three source repositories were newly cloned with SSH into the adjacent
`/user/miyapeng/rsi_evolve/references/`, outside the main project's Git tree.
Default branches were checked as `main`. The pinned tracked code is unmodified;
we import it in an explicitly selected Conda subprocess, without vendoring its
prompts, agent loops or graders. Model/data/VM downloads were not performed.

| Official repository | Exact source commit | License evidence | Files/functions called or audited |
| --- | --- | --- | --- |
| [zai-org/Vision2Web](https://github.com/zai-org/Vision2Web) | `d80a4a3d0ea7326541ad87314294269a633789e4` | README Dataset/License: **CC-BY-NC-SA-4.0, academic research only**. No root LICENSE file. `setup.py` has an inconsistent MIT classifier; do not treat that as resolved MIT permission. Referenced externally, no source copied. | `vision2web/core/{dataset,config,sandbox}.py`; `inference/engine.py`, `inference/adapters/*`, `inference/prompts.py`; `evaluation/{engine,functional_tester,visual_scorer,prompts}.py`; `analyze/analyzer.py`; `cli.py`, `docker/build.sh`, `setup.py`. New caller: `baseline/benchmarks/vision2web.py`. |
| [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld) | `8903ebeb0efdf74331bd7dc37ed91ccbbff22317` | Root `LICENSE`: Apache-2.0. | `evaluation_examples/test_all.json`, `test_nogdrive.json`, example JSON; `mm_agents/agent.py::PromptAgent`; `desktop_env/desktop_env.py::DesktopEnv`; `lib_run_single.py::run_single_example`; `lib_results_logger.py`, `lib_run_errors.py`; native provider modules, `run.py`, README and `requirements.txt`. New caller: `baseline/benchmarks/osworld.py`. This is the official Verified lineage, not a separate OSWorld-V2 checkout. |
| [arcprize/arc-agi-3-benchmarking](https://github.com/arcprize/arc-agi-3-benchmarking) | `353d38be471395b5ae47dbb44df95440677397e1` | Root `LICENSE`: MIT. | `main.py`; `benchmarking/{agent,base,swarm,model_config,model_configs.yaml,runtime_clients,recorder}`; `pyproject.toml`, `uv.lock`. New caller: `baseline/benchmarks/arc_agi_3.py`; calls native Swarm and scorecard, checks threaded-agent error exits. |

ARC's lock records SDK `arc-agi==0.9.9`; it was not installed during the initial
2026-11-04 adapter milestone and was installed for the 2026-11-05 smoke below.
The SDK's official [source/license](https://github.com/arcprize/ARC-AGI) and
[scorecard interface](https://github.com/arcprize/docs/blob/main/scorecards.mdx)
were inspected as reference (MIT; no SDK Git commit inferred from its package
version). We do not reimplement its scoring formula. The
[GUI-HARVEST README](https://github.com/GaryYang12345/GUI-HARVEST) was read only
to check comparison requirements; no code or prompt was copied. Its claimed
361-task suite differs from the pinned OSWorld lists here; see the runbook.

The local Vision2Web dataset and Conda environment from
`/user/miyapeng/Visual-Self-Verification` were reused for read-only discovery and
native-module import checks. That project's modified code, research methods,
private configuration and historical results were neither changed nor copied.
Its existing upstream archive has a different pin, so the worker explicitly
loads the newly pinned official checkout. See [OFFICIAL_BENCHMARKS.md](OFFICIAL_BENCHMARKS.md)
for exact asset paths, licenses, tests, environment requirements and NOT RUN items.


## Vision2Web image builder (2026-11-05 UTC)

The disposable Tencent build uses the maintained [osscontainertools/kaniko](https://github.com/osscontainertools/kaniko)
**v1.29.0-debug**, Apache-2.0. No builder source is copied into this repository.
The published multi-platform manifest is
`sha256:350929376b8a69afb1d8b362086c809e4ef6799fa1bcb317c2806ae1447b3845`;
its linux/amd64 manifest is
`sha256:b2ac9f31876a3176e8d1a8be6d03a455f79bdd026ca41a67aeac0af63cf88321`.
The publisher's manifest labels source revision
`632acb69ab000ddb12f99bf080e8f14dc7f2aa81`; this is image metadata, not a newly
cloned or audited builder-source checkout. The TCR mirror digest is independently
checked before use. The official Vision2Web Dockerfile is unchanged; its SHA256
is `ccfdef90c0f2cfaa79bbe5d976edbc56ebf3abea147e9e450e6fdb2ed58bf433`.
Credentials and the Tencent control plane are excluded from build snapshots.


The long-command transport diagnosis also inspected the user's existing
`minicpm-v5-agentic-rl/v5-2p6b-dual-harness/eval-code-text/ClawGym-Bench-e2b/`
`evaluation/localclawbench/internalbench/e2b_openclaw.py::_run_openclaw_agent`.
Its detached-process/status-file pattern informed the small foreground protocol;
no source, prompt, retry framework or agent loop was copied. E2B 2.62's installed
`commands/command.py` documents that `request_timeout` is ignored for streaming
commands; command timeout alone did not prevent the observed 150-second stream
failure. Model algorithms, prompts and grading remain upstream implementations.

## ARC online smoke and enterprise model profile (2026-11-05 UTC)

Reused the same clean official checkout at `353d38be471395b5ae47dbb44df95440677397e1`;
no new source repository or game/model dataset was downloaded. Its frozen
`uv.lock` runtime export was installed in Conda `rsi-arc3`. SDK package metadata
for `arc-agi==0.9.9` and `arcengine==0.9.3` declares MIT; no SDK source is vendored.
The installed `arc_agi/base.py` was inspected for anonymous-key registration,
online game access and scorecard closure. Runtime client and model-config loader
tests, native Anthropic runtime tests and Swarm tests were reused unmodified.

Original adapter additions: a process-scoped native `MODEL_CONFIG_PATH` setting,
profile-content hash, recorded Anthropic gateway endpoint, explicit use of
upstream `MAX_ACTIONS_HARD_CAP` only for smoke configurations, scorecard-log
credential redaction, and a private anonymous-key/catalog helper. Upstream
`BenchmarkingAgent`, prompts, text-grid observations, action parser, recording,
game API, time/action accounting and score formula remain unchanged. The custom
YAML model profile is configuration, not a copied or forked native agent.

The [official game-access docs](https://docs.arcprize.org/available-games) and
[official SDK](https://github.com/arcprize/ARC-AGI) informed anonymous public-game
access. A returned catalog does not establish play authorization for all games.
Measured results and environment/test commands: [ARC_AGI_3.md](ARC_AGI_3.md).


## OSWorld Tencent desktop provider (2026-11-05 UTC)

Reused clean xlang-ai/OSWorld `8903ebeb0efdf74331bd7dc37ed91ccbbff22317`.
New external SSH clone: [TencentCloudAgentRuntime/ags-cookbook](https://github.com/TencentCloudAgentRuntime/ags-cookbook),
default branch `main`, pin `b83d476e5e31006570cd8fab9a55ee342ca41491`, Apache-2.0
(root `LICENSE-Agent Sandbox Cookbook.txt`). No project-owned unlicensed source
is copied. The cookbook's OSWorld overlay targets an older base, so its full
DesktopEnv is not overlaid on the current pin.

Reused through per-episode copies: `examples/osworld-ags/overlay/OSWorld/desktop_env/providers/ags/`
`provider.py`, `manager.py`, `config.py`, `sandbox_setup.py`, `cdp_proxy.py`,
`socat_wrapper.sh`. Reused its authenticated HTTP/WebSocket proxies and guest
CDP bridge; a small `osworld_ags_stream.patch` changes buffered successful HTTP
responses to streaming without changing payloads. A separate
`osworld_screenshot.patch` makes the read-only screenshot idle timeout explicit
(60 seconds for the measured Tencent link), leaving action execution unchanged. Overrode lifecycle to use E2B 2.62, verify tool identity, propagate
cleanup failure and replace no-op snapshot reset with fresh-instance creation.
The official cookbook tests for sandbox bootstrap and CDP proxy are run directly.

Modified-at-runtime copy: OSWorld `desktop_env/desktop_env.py` constructor only,
through the versioned `baseline/benchmarks/osworld_provider.patch` (Apache-2.0;
license retained in `licenses/OSWorld-Apache-2.0.txt`). Original reset, step,
recording checks, task setup and evaluators remain unchanged. Original adapter
`osworld_gateway.py` subclasses native `PromptAgent` only at `call_llm`, retaining
its prompts/history/parser. `requirements/osworld-eval.in` selects the official
evaluator dependencies for a separate CPU host environment, excluding unused
providers and agents. No Slime/training settings are changed.

The [Tencent OSWorld beta guide](https://cloud.tencent.com.cn/document/product/1814/129382)
was checked for the native API/tool-name protocol and official cookbook link.
Actual template and run evidence: [OSWORLD_TENCENT.md](OSWORLD_TENCENT.md).

`osworld_recording_guest.patch` adds a hash-guarded Tencent modular-launcher case
to native `recording_update/guest.py`. Official discovery, installer transaction,
rollback and recorder handlers remain unchanged. The inspected Tencent launcher,
server and HTTP module hashes are recorded in the patch. Template source was read
inside disposable user-authorized sandboxes for compatibility diagnosis; it is
not vendored. Its original action/setup routes remain in place. Runtime copies
and patches are recorded per episode; both Git checkouts stay clean.

## SWE-bench Multimodal v2 adapter (2026-11-05)

Reused the existing clean SSH checkout of [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench),
default branch `main`, exact pin `87ab1f6ced28f75ba73ca899dc759b019310944a`
(release 5.0.1), MIT. No new repository clone and no update of the training
installation's `swebench==4.1.0`. A source-path-isolated evaluation worker imports
v5. The source license is retained in `licenses/SWE-bench-MIT.txt`.

Reused unchanged: `swebench/harness/utils.py::make_test_spec`,
`harness/grading.py`, `harness/log_parsers/javascript.py`, `types.py` and their
upstream tests. `swe_mm_transport.patch` adds optional explicit transport
injection to a per-episode copy of `harness/run_evaluation.py::run_instance` and
its binary-asset copy helper. Patch application order, asset restoration order,
eval scripts, parser selection and scoring remain native. Default Docker code
paths remain in that copy; the adapter supplies Tencent creation/copy/exec/cleanup.
Missing private image assets are additionally classified as invalid evaluation,
not ordinary task failure. The reference checkout is never patched in place.

Existing project `baseline/runtime.py::run_agent`, `harnesses/seed`,
`baseline/sandbox.py::TaskSandbox` and `baseline/tencent.py::TencentSandbox` are
reused. Original adapter code adds public issue-image API binding and restricted
Node PATH setup. Live AGS commands lacked the official Docker `ENV` (notably
`NVM_DIR`), so the adapter reads `Config.Env` from the pinned OCI digest and
restores it for trusted setup/grading; no host environment or credentials are
forwarded. Agent commands still use the original cleared environment plus Node
PATH. It does not copy another agent or introduce a new training loop.

Pinned data: `SWE-bench/SWE-bench_Multimodal` at
`4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5`, actual dev/test sizes 100/480.
The dataset card has no explicit license metadata; do not infer that source MIT
covers screenshots or task repository content. Data is downloaded to existing
ignored asset storage, not redistributed. Old dataset-card narrative counts are
recorded as stale in [SWE_MM.md](SWE_MM.md).
