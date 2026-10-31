# Third-party notices

This is a Git worktree derived from Microsoft/MSR-Orchard Self-HPO at the commit listed in `docs/UPSTREAM.md`. Its original MIT [LICENSE](LICENSE) and upstream source history remain intact. The extracted grading functions and adapted H-step scripts remain covered by that license and their original notices.

The retained mini-SWE-agent fork is covered by [mini-swe-agent/LICENSE.md](mini-swe-agent/LICENSE.md) (MIT). The historical bundled Slime is covered by [slime/LICENSE](slime/LICENSE) (Apache-2.0). Active training imports the separate THUDM/slime dependency (Apache-2.0), with its license retained in that checkout. Native Slime test fixtures are imported from that dependency only during tests.

E2B SDK 2.62.0 is an Apache-2.0 dependency, not vendored. Tencent transport adapts MIT code from the earlier local prototype using the supplied integration protocol. Orchard-Agentic (MIT) remains an unchanged historical reference, no longer an active requirement. SWE-bench_Pro-os (MIT, Scale AI) remains separate and unchanged. SWE-bench 4.1.0 package notices are retained. Model weights and task repositories are not redistributed; their individual terms still apply. Public split metadata does not relicense dataset contents or task code.

Exact commits, source paths and modification types: [UPSTREAM.md](docs/UPSTREAM.md).

The adjacent NVIDIA/Megatron-LM training source is BSD-3-Clause by default, with
additional per-file third-party notices retained in its full LICENSE. The two
applied patches come unchanged from the pinned Apache-2.0 Slime dependency.
Neither dependency is vendored into this worktree; pins and patch hashes are
recorded in `docs/UPSTREAM.md`.
