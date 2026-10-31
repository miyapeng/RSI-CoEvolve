# Upstream provenance

| Repository | Pinned commit | License | Use / modification |
|---|---|---|---|
| https://github.com/MSR-Orchard/Self-HPO | 3c3d0bd93a8a5a72399c855783460b141ea27d49 | MIT (LICENSE); mini MIT (mini-swe-agent/LICENSE.md); bundled Slime Apache-2.0 | Main worktree. Seed `harnesses/seed`, mini agent/actions/submission semantics retained. H-step experience/parent/fire/record scripts reused; path/config integration changes recorded in Git. Source-specific P-step grader functions extracted from RL imports, semantics retained. Original 35B shell launcher is historical and not active. |
| https://github.com/THUDM/slime | 0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e (v0.4.0) | Apache-2.0 | Existing adjacent dependency, unchanged: scripts/models/qwen3.5-9B.sh, slime/agent/adapters, trajectory capture, Megatron GRPO, HF checkpoint saver. |
| https://github.com/microsoft/agent-lightning | d381995396274039f2bb1cbe5ff42ac8067f4e47 | MIT | Inspected existing asset only; not imported or maintained as another trainer. |
| Local harness-policy-coevolution | 937b5e8 | MIT, upstream notices retained there | Reuse dataset/model links and lessons from native Slime adapter integration. Research algorithms not imported. |

| https://github.com/microsoft/Orchard-Agentic | bd8da32b3a075128700caab75bbc491fb6fdc16e | MIT | Only newly cloned repository, using SSH. Adjacent editable SDK `orchard_env/`, SandboxClient create/exec/network/delete; source unchanged. |
| https://github.com/scaleapi/SWE-bench_Pro-os | 66f92766bba642462d4bbe5479e83f91f9211862 | MIT, Copyright 2026 Scale AI | Existing adjacent dependency; Pro V2 per-task tests used by the original mini verifier. No tests copied or modified. |

Default branches observed: all five repositories above use `main` (Slime checked with `git ls-remote --symref`, remote HEAD then `540ea919e45b8b929198abce008a103be9cfc779`). Slime remains pinned to the existing local v0.4.0 checkout, not that newer HEAD. This worktree shares Self-HPO Git history; dependency repositories remain outside this worktree's commits. Original root MIT, mini MIT and bundled Slime Apache-2.0 licenses are preserved. See [third-party notices](../THIRD_PARTY_NOTICES.md).

## Actual file-level reuse and adaptations

| Upstream path | Local use/change |
|---|---|
| `harnesses/seed/{agent.py,config.yaml}` | Unchanged starting harness and prompt. Seed aliases `ProgressTrackingAgent`; no additional loop. |
| `mini-swe-agent/src/minisweagent/{agents/default.py,run/benchmarks/swebench.py,models/utils/actions_toolcall.py,environments/docker.py}` | Unchanged dependency calls: `run`, `step`, `execute_actions`, marks, formatting and submission detection. `baseline/runtime.py` supplies model/environment protocols, not another agent loop. |
| `scripts/h_step/propose_prompt.tmpl` | Original file unchanged. `baseline/search.py` renders it and appends the explicit API/code-edit contract and bounded optimization-only evidence. Original ADD-on-parent restriction retained for this baseline. |
| `scripts/h_step/build_experience.py`, `parent_check.py` | Only configurable working root (`BASELINE_WORKDIR`); retain original diagnosis and parent-retention check. |
| `scripts/h_step/fire_gate.py` | Configurable root; accepts already collected real gate trajectories; discovers candidate class name. Existing wiring/action/submission checks retained. |
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
| mini `swebench_verify_orchard.py`, `swebench_multilingual_verify_orchard.py`, `swebench_pro_verify_orchard.py` | Unchanged final graders, separate process entrypoint and frozen dataset versions. Pro V2 references official tests at the pin above. |

No HarnessAdapt-RL proprietary source or prompts copied. No Meta-Harness/RRSI/AHE engine merged. No Agent Lightning/veRL training code imported. Official URLs, local licenses and pins were checked; no global dependency upgrade performed. The existing old prototype contributed local assets and prior adapter lessons, not its Item/trigger/pruning algorithms.
