# Minimal Benchmark Adapter foundation

**Follow-up implementation (2026-11-04):** three concrete official evaluation
adapters now exist. [OFFICIAL_BENCHMARKS.md](OFFICIAL_BENCHMARKS.md) documents the
native entry points, pins, setup and actual verification. The foundation audit
and measured results below describe the earlier interface-only milestone;
its statements about no external adapters do not describe the follow-up.
SWE H/P boundaries and historical resume hash enforcement remain unchanged.

Implemented in `/user/miyapeng/rsi_evolve/self-hpo-baseline`, the worktree whose
`rsi` remote is `git@github.com:miyapeng/RSI-CoEvolve.git`. The starting revision
was `d649f79`; the worktree was clean before this change. The older
`harness-policy-coevolution` worktree is not modified. This milestone adds task
loading/execution/verification delegation, not a multimodal H/P training system.

## Audit of the existing code

| Responsibility | Actual code and coupling | Treatment in this milestone |
|---|---|---|
| Benchmark-specific data | `data.py`: Self-HPO JSONL, Scale-SWE / SWE-rebench source rules, repository/patch grouping, frozen split fingerprints, `metadata.instance_id`, explicit task subsets | Keep validation and split algorithms. Move subset selection into `SWEBenchmark.load_tasks`; retain `data.configured_tasks` as a compatibility entry point. |
| Benchmark-specific verification | `reward.py`, `grading_spec.py`, `grading_report.py`: source-specific test construction, patch-application ladder, trusted test execution, resolved score | Delegate unchanged. SWE still maps solved to +1, unresolved to -1. Incomplete grading still raises, with null error score. |
| Environment/sandbox | `tencent.py` handles AGS/E2B transport and pinned registry images; `sandbox.py` also has SWE-specific Git reset/history isolation, unprivileged shell setup and grader preparation | Retain both. A future adapter may use a different environment; generic dispatch does not instantiate Tencent. |
| Agent runtime | `runtime.py`: mini-SWE-agent tool-call serialization, shell actions, step/timeout limits, submission and trajectory recording | Retain the one SWE agent loop. Other adapters own their runtime rather than being forced through mini-SWE-agent. |
| Harness evolution | `search.py`: Self-HPO success/failure diagnosis, offline optimization-only gold patches, Python hook edits, parent/fire gate, paired promotion | Keep unchanged selection semantics. Explicitly SWE-only until another domain has its own diagnosis/edit/validation contract. |
| Policy training | `training.py`: Dense 9B settings and text task packaging. `slime_rollout.py`: Qwen parsers, OpenAI capture, native segment masks/logprobs, trajectory-level GRPO normalization | Keep optimizer, loss and capture logic. Frozen task lookup now delegates to the adapter using the exact saved P-step task IDs. Non-SWE training fails explicitly. |
| Experiment management | `outer_loop.py`, `records.py`, `report.py`: fixed H→P schedule, pair/snapshot publication, cross-evaluation, stage resume | Schedule unchanged. Change only the task-selector import and include nested adapter source files in the resume code hash. |
| Serving / CLI / final benchmarks | `serving.py`, `config.py`, `audit.py`, `__main__.py`, `final_eval.py`: Qwen/SWE configuration, GPU and Tencent checks, explicit oracle/final-eval commands | Existing SWE CLI remains valid. General CLI/service preflight and independent final benchmark adapters are not generalized here. |

The important existing connection is that **both H-step evaluation and native
Slime generation call `execution.rollout()`**. The adapter sits at this shared
boundary. `training.prepare()` still writes text-only Slime prompt records, so
routing an episode through an adapter does not automatically make it trainable.

## Interface and ownership

`baseline/benchmarks/base.py` defines a small structural `BenchmarkAdapter`
protocol and record types. No inheritance from a new agent class is required.

| Method | Input / output and responsibility |
|---|---|
| `load_tasks(config, role, limit)` | Load benchmark-owned task records from the configured frozen manifest/selection. Records may be dicts or domain objects. Preserve split membership and order. |
| `task_info(task)` | Return `TaskInfo(task_id, dataset_source, split)` for generic task identification and seed assignment. IDs must be stable, unique in their split, and safe as local path components. |
| `public_task(task)` | Explicit allowlisted projection for the agent. Its shape belongs to the benchmark; SWE retains its exact old projection. |
| `run_episode(client, task, checkpoint, harness, config, output, seed)` | Own runtime/environment lifecycle, give only public information to the agent, persist trajectories, invoke verification separately, and return an `EpisodeResult`. No universal model client or action grammar is imposed. |
| `verify(task, episode, runtime_config, output)` | Trusted verifier receives the private task plus submitted episode output. Return score, optional training reward, and private verifier details. This method is also callable independently for diagnostics. |

`EpisodeResult` names the common fields already used by SWE: task/source/split,
policy reference, harness revision, sampling parameters, trajectory path,
termination/error, score/reward, tokens, tool calls, latency and `simulated`.
An adapter may preserve domain fields such as SWE's `patch`, `sandbox_id` and
`sandbox_image`. Unmeasured quantities remain `None`. `execution.rollout()` adds
`benchmark` and writes the completed `result.json`; `evaluate()` retains the old
seed calculation, repetition paths and `results.jsonl` behavior.

The protocol is a **trusted Python boundary**, not a security sandbox or a
runtime schema validator. Adapter authors must keep private task records,
answers and `Verification.details` away from model inputs/tool observations.
Runtime-specific exceptions must preserve useful failure records and cleanup;
infrastructure failures must not be converted into failed-task rewards. The SWE
adapter retains the existing `error.json` and early `sandbox.json` behavior.
Host-side experiment/verifier records are not agent-visible observations.

`baseline/benchmarks/__init__.py` contains an explicit local name-to-class table
and `get_benchmark(config)`. Omitting `benchmark` selects `swe`; explicitly setting
`benchmark: swe` has the same execution behavior. Unknown names raise rather than
silently falling back. To add another local adapter, add its module under
`baseline/benchmarks/` and its name in this table. There is no plugin loader,
provider autodetection, dependency download, or external code discovery.

## Observation and artifact records

The optional `Observation` convention supports `text`, `image`, `state`, and
`tool_result`. It is a recording format, not a GUI action specification or a
mandatory replacement for existing benchmark message formats. For example:

```json
{
  "kind": "image",
  "artifact": {
    "path": "artifacts/screen-0001.png",
    "media_type": "image/png",
    "metadata": {"width": 1280, "height": 720, "sha256": "..."}
  }
}
```

References point to stored files, with dimensions/time/hash when available.
Relative references should be interpreted against the episode directory;
absolute references are also supported for local fixtures. No image bytes or
large base64 strings are required in JSON. The future runtime is responsible
for loading an artifact and producing the model provider's actual input format.
Artifact path/metadata persistence, retention and access permissions remain the
adapter/environment owner's responsibility.

A **container image** is separate: SWE's legacy public `image` key and result
`sandbox_image` identify task container images; they are never visual artifacts.
The new observation form always places visual file references under `artifact`
with an image MIME type. Existing SWE records are not renamed or migrated.

## Existing SWE mapping

`baseline/benchmarks/swe.py` delegates to:

- `data.tasks` and the unchanged `data.public_task` for source checks, frozen
  membership and public task projection;
- `sandbox.open_task` for the same prepared Tencent task sandbox;
- `runtime.run_agent` for the same actual mini loop and candidate methods;
- `reward.grade` for the same separate original verifier and reward mapping.

The old body of `execution.rollout` is moved into this adapter; no second agent
loop is introduced. Successful records gain the additive `benchmark: swe`
field. Existing configuration, command syntax, patch fields, score semantics,
model messages, tool actions and candidate promotion rules are retained.

Slime still verifies the train-file hash, split-manifest hash, allowed task IDs
and frozen harness hash. It loads the exact IDs in the **saved P-step selection**
through the adapter, rather than assuming the current default task prefix. This
also preserves the existing `--task-id` / explicit `prepare()` selection behavior.
Native rollout capture, masks/logprobs, optimizer and GRPO code are untouched.

## CPU-only second adapter

`tests/baseline/test_benchmarks.py` defines a test-only `FixtureBenchmark` with
an opaque dataclass task, a text instruction, an SVG image artifact reference,
structured state and a tool-result record. Its separate deterministic callable
returns a literal answer; it never interprets the image. Its verifier compares
that answer with private expected data. Records explicitly say `simulated=True`
and unknown token/tool/latency metrics are null.

The tests install that adapter only in the temporary test registry and exercise:

- generic selection and repeated `execution.evaluate()` calls with stable seeds;
- positive and negative verifier results;
- absence of hidden evaluator metadata in agent inputs and episode JSON records;
- a two-round **Base-mode** outer loop using the actual `evaluate()` and a fake
  serving client, followed by idempotent resume;
- refusal to enter SWE-specific H-step, P-step or CLI preflight for this adapter.

This demonstrates task-loading/execution/grading integration without changing
core orchestration. It does **not** simulate successful RL or claim a non-SWE
H/P loop. Existing CPU tests separately retain SWE H/P schedule/resume coverage.
The fixture is not a production adapter or benchmark result.

## Future GUI / Visual Coding integration and current limits

A later integration can add an adapter that loads its own task objects, exposes
public text/screenshot/state references, runs its native browser/GUI/coding
agent in its own environment, and uses its official verifier. Basic callers
already use adapter selection and normalized episode records. The adapter does
not need to emit bash commands or mini-SWE-agent submissions.

The following are deliberately still needed before a true multimodal H/P run:

1. A real runtime, tool/action schema and environment lifecycle, with reliable
   screenshot/file capture and public/private evaluation isolation.
2. Domain-specific harness-edit boundaries, trajectory diagnostics and gates.
   The current Python hook validator, gold-patch diagnosis and fire gate are SWE
   assumptions, not general GUI mechanisms.
3. Multimodal model/processor input serialization and native Slime support for
   image inputs, captured tensors, token alignment, masks/logprobs, batching and
   gradient flow. The current trainer learns the text language model; retaining
   frozen vision weights during Dense export is not multimodal RL support.
4. Appropriate model serving, resource preflight and configuration, plus harness
   snapshot coverage for that runtime's files. Current `Operations.serve`, CLI
   setup, checkpoint checks and snapshot filenames remain tied to the SWE recipe.

The protocol intentionally does not claim those capabilities. `require_swe`
rejects unsupported production H/P and CLI-preflight paths before resource use.
The fixture supplies a serving callable through the existing test-only Operations
boundary. Completing generic service/CLI selection and domain-specific H/P
integration would be a separate milestone, not a safe small change here.

Recommended next step: choose one future domain's task/public-input/verifier
contract and implement a single-task inference-only smoke with its own runtime
and file artifacts. Validate scoring and information isolation before connecting
that runtime to policy training. No adaptive scheduling or new loss is needed.

## Resume and validation

`outer_loop.core_hash()` now recursively hashes Python sources in `baseline/`,
including the new adapter modules. A test changes an adapter source file in an
isolated temporary source tree and confirms resume rejection. Existing
config/data/checkpoint/harness checks are retained. **Historical runs may no
longer resume under this changed source.** Use their original code revision;
never edit saved hashes or weaken checks to force compatibility. Adding or
editing a local adapter also changes the code hash, even if another adapter is
selected. External runtime dependencies still need explicit version pinning.

Actual verification commands and results for this milestone are recorded below.
Historical GPU success in `STATUS.md` belongs to its original code revision and
is not a GPU rerun of this change. No GPU training, paid API call, remote sandbox,
model/dataset download, external benchmark integration or remote Git push is
part of this milestone.

```bash
cd /user/miyapeng/rsi_evolve/self-hpo-baseline
CONDA_PREFIX=/user/miyapeng/miniconda3/envs/self-hpo-baseline \
  PATH=/user/miyapeng/miniconda3/envs/self-hpo-baseline/bin:$PATH \
  bash scripts/baseline/test_cpu.sh

# CPU-only native adapter checks in the existing dependency-rich Conda prefix:
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 MSWEA_SILENT_STARTUP=1 \
  /user/miyapeng/miniconda3/envs/self-hpo-gpu/bin/python \
  -m pytest tests/baseline/test_native_slime.py -q
```

Measured results on 2026-11-04:

- Baseline tests: **38 passed, 1 skipped**. The skipped native BatchBuilder check
  needs Ray, absent from the CPU development environment.
- Unmodified upstream Slime agent tests: **49 passed, 1 skipped**.
- Native Slime adapter/group normalization tests in the existing GPU-dependency
  Conda prefix, with GPUs hidden: **2 passed**. Pytest emitted one existing
  unknown `asyncio_default_fixture_loop_scope` configuration-option warning;
  no package was installed/upgraded to suppress it.
- CPU `pip check`, Ruff, Python compilation and `git diff --check` passed.
- Offline wheel build from a temporary source copy (`--no-index --no-deps
  --no-build-isolation`) verified inclusion of all three new benchmark modules.
  This checks package inclusion, not a full standalone GPU installation.

No historical run, split manifest, model weight or upstream dependency source
was modified. No remote Git operation or real task run was performed for this
milestone. The new code's real GPU regression is **NOT RUN** by task scope;
previous GPU results remain historical evidence only.

## SWE-MM extension (2026-11-05)

`SWEMultimodal` adds an explicit official dev/test mapping via
`OfficialBenchmark.evaluation_role(config)`; all pre-existing official adapters
keep the original `final_test` default. No H/P scheduler or training dispatch
changed. SWE-MM uses the existing mini agent with an image-aware API boundary and
a copy of the official SWE-bench v5 grader with an explicit Tencent transport.
Public issue images and private test/reference images remain separate. See
[SWE_MM.md](SWE_MM.md) for source/data pins, run commands and actual validation;
this is evaluation support, not multimodal policy training.
