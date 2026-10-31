# Upstream provenance

| Repository | Pinned commit | License | Use / modification |
|---|---|---|---|
| https://github.com/MSR-Orchard/Self-HPO | 3c3d0bd93a8a5a72399c855783460b141ea27d49 | MIT (LICENSE); mini MIT (mini-swe-agent/LICENSE.md); bundled Slime Apache-2.0 | Main worktree. Seed `harnesses/seed`, mini agent/actions/submission semantics retained. H-step experience/parent/fire/record scripts reused; path/config integration changes recorded in Git. Source-specific P-step grader functions extracted from RL imports, semantics retained. Original 35B shell launcher is historical and not active. |
| https://github.com/THUDM/slime | 0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e (v0.4.0) | Apache-2.0 | Existing adjacent dependency, unchanged: scripts/models/qwen3.5-9B.sh, slime/agent/adapters, trajectory capture, Megatron GRPO, HF checkpoint saver. |
| https://github.com/microsoft/agent-lightning | d381995396274039f2bb1cbe5ff42ac8067f4e47 | MIT | Inspected existing asset only; not imported or maintained as another trainer. |
| Local harness-policy-coevolution | 937b5e8 | MIT, upstream notices retained there | Reuse dataset/model links and lessons from native Slime adapter integration. Research algorithms not imported. |

Orchard SDK pin and final modified-file inventory are recorded after acquisition. No HarnessAdapt-RL proprietary source or prompts copied. No Meta-Harness/RRSI/AHE engine merged. Official repository URLs and local pins were checked; no global dependency upgrade performed.
