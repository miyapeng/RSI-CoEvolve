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
