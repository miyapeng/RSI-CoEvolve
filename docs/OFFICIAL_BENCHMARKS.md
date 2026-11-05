# Official benchmark adapters

Implemented in `self-hpo-baseline`, the worktree associated with
`git@github.com:miyapeng/RSI-CoEvolve.git`. The older
`harness-policy-coevolution` prototype is unchanged. This milestone adds native
**evaluation** paths, not multimodal GRPO or new harness-search algorithms.

## Native implementations and boundaries

| Adapter / module | Official execution and verification | Environment |
| --- | --- | --- |
| `vision2web` / `baseline/benchmarks/vision2web.py` | `DatasetManager`, `InferenceEngine.run_single_project`, `EvaluationEngine.evaluate_single_project`, `ResultAnalyzer.analyze`; default official Claude Code harness | Official Vision2Web Docker image, generated website, Playwright CLI; separate functional and visual judge models |
| `osworld_verified` / `baseline/benchmarks/osworld.py` | `PromptAgent`, `DesktopEnv`, `lib_run_single.run_single_example`; the official driver calls `env.evaluate()` | Local Docker/KVM, VMware or VirtualBox; opt-in Tencent OSWorld desktop template, validated for one real task |
| `arc_agi_3` / `baseline/benchmarks/arc_agi_3.py` | Official `BenchmarkingAgent` through `Swarm.main`; SDK/server scorecard | Official online ARC game API and a pinned native model profile |

Upstream agent loops, prompts, action parsers and verifiers remain in pinned
dependencies. The Tencent OSWorld worker makes a per-episode source copy for
explicit provider, screenshot-transport and recorder compatibility patches.
The OSWorld single-example function is also used by its ordinary parallel runner;
we call it directly for one bounded episode and close the environment in `finally`.
ARC retains the official profile's state management; its separate bounded smoke
config explicitly uses the upstream development action cap.
Different native action spaces remain different. Tencent AGS/E2B continues to
serve the working SWE baseline. Vision2Web now has an opt-in
[Tencent command transport](VISION2WEB_TENCENT.md), with a real website/browser/
visual-scoring episode recorded on 2026-11-05 (no functional cases in that task).
[ARC online inference](ARC_AGI_3.md) also completed an eight-action public-game
smoke and official scorecard. ARC needs no Tencent sandbox in this online path;
OSWorld now reuses the supplied Tencent OSWorld desktop template through its
[Tencent adapter](OSWORLD_TENCENT.md). A SWE command container is not a desktop VM.

The existing adapter registry dispatches these names. `official.py` adds only
task identity, a clean pinned-source check, a subprocess boundary and common
records. `official_worker.py` runs with the selected Conda Python and explicitly
prepends the pinned source directory, avoiding another project's editable install.
`python -m baseline.benchmarks` is an independent evaluation CLI; it uses the
existing `execution.evaluate` without starting SWE SGLang or changing `outer_loop`.
`list`, `plan` and `preflight` do not call a model or create a sandbox. `run` is
explicit and fails preflight before launching if required resources are absent.

These adapters accept `final_test` tasks only. They reject use as optimization,
development or audit training pools; existing `require_swe` checks still reject
non-SWE H/P entry points. No final benchmark has been placed in the SWE splits.
Native API model names identify the actual configured model; they are not
presented as exported/reloaded Qwen checkpoints. The recorded sampling seed is
an experiment identifier; `seed_applied_to_model: false` states that these native
APIs do not receive it. Repeated online runs need not be deterministic.

## Private inputs, scoring and records

- Vision2Web exposes requirements, prototype image file references and public
  resources. Its official inference copier excludes `workflow.json`; only the
  host evaluation engine loads that file. VS and FS remain separate native
  analysis fields, with `score: null` and `training_reward: null`; no combined
  reward is invented. Native evaluation failure status and its analyzer's
  zero-fill behavior are preserved in `reward.analysis` / `reward.evaluation`.
  A native `success` status is not proof that every browser workflow succeeded;
  inspect individual case files and logs, as upstream catches some workflow errors.
- OSWorld passes the full example to `DesktopEnv.reset` and only instruction
  plus environment observation to `PromptAgent.predict`. Evaluator/config
  definitions stay on the host. The scalar is the original `env.evaluate()`
  result, including fractional scores. Screenshots and recordings stay as files.
- ARC accepts explicit full game IDs, never silently expands a prefix or runs
  all games. Observations arrive from the official game environment; no game
  implementation/solution is passed to the model. We retain the full official
  scorecard and its native score scale, without substituting win rate. SDK
  scorecard `api_key` is excluded from our saved copy. Agent/API error exits,
  including failed upstream worker threads, raise instead of becoming valid scores.

Each fresh episode records `request.json`, `public_task.json`, `environment.json`,
`official.log`, native artifacts, `native_result.json`, `verification.json` and
the shared `result.json`. Environment metadata contains package versions and
source pins, not environment values. API credentials are referenced by variable
name, never placed in commands or adapter request JSON. Native logs and generated
projects are private research artifacts and remain under ignored `runs/`.
Visual metadata uses file paths, media types and SHA256, not base64 JSON payloads.

Model identity, source commit/options hash, adapter source hash and task
fingerprint are bound to each record. Vision2Web hashes all files in the selected project at planning/execution;
OSWorld hashes its example and records suite hash, while its externally fetched
setup assets still require separate provenance. ARC fingerprints the requested
game ID; this is **not** a hash of inaccessible server-side game code. Native
scorecard/game IDs and package versions are needed for reproduction.
Existing episode/output directories are rejected to prevent upstream completion
skips from turning stale scores into new results. Failed workers produce an
explicit `failure.json` with null score and retain logs; there is no automatic retry.

## Sources, local assets and version differences

Pins, licenses and exact used files are in [UPSTREAM.md](UPSTREAM.md).
The three official source repositories were cloned over SSH into the adjacent
`../references/`, outside the project commit. Their tracked source is unchanged.

Local Vision2Web assets were discovered and reused read-only:

```text
/user/miyapeng/Visual-Self-Verification/data/vision2web/extracted
/user/miyapeng/miniconda3/envs/vision2web/bin/python
```

The data contain 100 webpage, 66 frontend and 27 website tasks. The existing
dataset manifest records HF revision `8f03299d92b9bd852e93852d0c21e8a4848ab661`.
Its archive checksums were not independently revalidated in this milestone.
The existing Conda package resolves to another project's older upstream archive
(`577f9397b3db8fc6d828adde254a830caa65d515`); we leave that project/environment
unchanged and import our pinned checkout explicitly. This import path was tested.

**OSWorld-Verified is not OSWorld-V2.** At our pinned official OSWorld commit,
`test_all.json` has **368** tasks (SHA256
`47c54203369bd438abf279f6b9836bffc5aa43de6cbaccdfed546993f3fe9149`),
and `test_nogdrive.json` has **360**. The
[GUI-HARVEST README](https://github.com/GaryYang12345/GUI-HARVEST)
describes a **361-task** no-GDrive universe and requires a separately installed
seed harness, for example Agent S3. Do not directly compare numbers from our
PromptAgent baseline to that experiment. Before comparison, align their exact
source/task membership, seed harness/model, observations/actions, steps, repeated
rollouts and splits. `official.suite` chooses between the two pinned official
lists; it does not repair that historical task-count difference. GUI-HARVEST
source and Agent S3 were not imported into this project.

## Conda setup and commands

Keep benchmark environments separate from the working Slime training environment.
Only source fetching is automatic. The initial 2026-11-04 adapter milestone
installed no environments; the 2026-11-05 ARC smoke subsequently installed a
dedicated Conda environment from the official frozen lock. See its linked runbook.

```bash
cd /user/miyapeng/rsi_evolve/self-hpo-baseline
conda activate self-hpo-baseline
python scripts/baseline/fetch_dependencies.py --ssh \
  --benchmarks Vision2Web OSWorld arc-agi-3-benchmarking
python scripts/baseline/fetch_dependencies.py --check \
  --benchmarks Vision2Web OSWorld arc-agi-3-benchmarking

# On a new machine only; reuse existing Conda environments when possible.
conda create -n rsi-vision2web python=3.12 pip
conda run -n rsi-vision2web python -m pip install -e ../references/Vision2Web
# OSWorld: use environment-osworld.yml and the CPU lock in OSWORLD_TENCENT.md.
# ARC: use environment-arc3.yml and the frozen-lock commands in ARC_AGI_3.md.
```

OSWorld's focused dependency profile is frozen in `requirements/osworld-eval.lock`
and tested in Conda `rsi-osworld`; its upstream requirements contain unbounded
entries. ARC's frozen `uv.lock` export was installed inside `rsi-arc3`
(including `arc-agi==0.9.9`); plain pip installation of its `pyproject` would not
enforce that lock. The worker imports its pinned source directly. Per-episode
package inventories are recorded. The SWE CPU and GPU environments were not upgraded.

Start with read-only commands:

```bash
export VISION2WEB_DATA=/user/miyapeng/Visual-Self-Verification/data/vision2web/extracted
python -m baseline.benchmarks list --config configs/benchmarks/vision2web.yaml --limit 3
python -m baseline.benchmarks plan --config configs/benchmarks/vision2web.yaml \
  --limit 1 --output runs/vision2web-first
python -m baseline.benchmarks preflight --config configs/benchmarks/vision2web.yaml \
  --python /user/miyapeng/miniconda3/envs/vision2web/bin/python \
  --output runs/official-adapter-checks/vision2web

python -m baseline.benchmarks list --config configs/benchmarks/osworld_verified.yaml --limit 3
python -m baseline.benchmarks plan --config configs/benchmarks/osworld_verified.yaml \
  --limit 1 --output runs/osworld-first
python -m baseline.benchmarks preflight --config configs/benchmarks/osworld_verified.yaml
python -m baseline.benchmarks preflight --config configs/benchmarks/arc_agi_3.yaml
```

Preflight exits **2** for missing requirements. It is offline: an available
credential variable is not a validated key, an import is not an API/VM smoke,
and a local image is not a successful browser deployment.

For an explicit later live run, fill model endpoints/credential environment
references in a local config. Vision2Web needs the official Docker image built
using its `docker/build.sh`, a native coding-model endpoint and both judge
endpoints. Its native analyzer currently requires a model alias without `/` or
`:`; the adapter fails clearly rather than silently renaming the model. OSWorld
needs an existing official Ubuntu VM image in `official.path_to_vm`, a matching
native model credential and the selected virtualization provider; local images
avoid the default VM download path. ARC needs `ARC_API_KEY`, the selected native
model-profile credentials and full available game IDs. List games deliberately
in its official environment via `main.py --list-games`; this contacts the server.
The sample ARC profile has substantial native budgets, not a cheap toy rollout.

```bash
# Only after resources and intended API spending are arranged:
python -m baseline.benchmarks run --config configs/benchmarks/vision2web.yaml \
  --python /user/miyapeng/miniconda3/envs/vision2web/bin/python \
  --task-id webpage/abc --limit 1 --output runs/vision2web-live-001

# Configure official.path_to_vm first and point --python at the OSWorld Conda Python.
python -m baseline.benchmarks run --config configs/benchmarks/osworld_verified.yaml \
  --python /path/to/conda/envs/rsi-osworld/bin/python \
  --limit 1 --output runs/osworld-live-001

python -m baseline.benchmarks run --config configs/benchmarks/arc_agi_3.yaml \
  --python /path/to/conda/envs/rsi-arc3/bin/python \
  --task-id FULL_GAME_ID_FROM_OFFICIAL_API --limit 1 --output runs/arc-live-001
```

Model/provider settings remain in the three `configs/benchmarks/*.yaml` files.
No key was copied from the conversation into source/config. Existing secrets
were not printed, and no cluster task, Docker daemon change or remote push occurred.

## Validation and next step

Measured on 2026-11-04:

- Five new CPU tests passed. They exercise official Vision2Web input copying,
  inference control flow and ResultAnalyzer with fake container/API boundaries;
  the real OSWorld single-example driver with a fake desktop/model/verifier;
  ARC profile binding, official-call contract, scorecard retention and failed
  thread detection using stubs. These fixture scores are not benchmark results.
- Project CPU suite: **43 passed, 1 skipped**; unchanged upstream Slime agent
  suite: **49 passed, 1 skipped**. Missing Ray accounts for the local project skip.
- Native Slime adapter checks in the existing GPU-dependency Conda environment,
  with `CUDA_VISIBLE_DEVICES=''`: **2 passed**; one existing pytest configuration
  warning. No GPU workload ran. Use the repository's CPU test command with the
  Conda `bin` directory on `PATH`: a direct Python-only invocation failed the
  existing image-import test because it could not find Conda's `rsync`.
- CPU dependency consistency, Ruff and Python compilation passed. Fixed source
  checks passed for all three newly cloned repositories and existing Slime.
- Vision2Web local data discovery and imports from the pinned official checkout
  passed using the existing Conda environment. No Docker executable was found;
  coding/judge credential variables were absent in the checked environment.
- OSWorld discovery passed; its Python dependencies, local VM, KVM and Docker
  prerequisites were absent. ARC lacked native dependencies, credentials and
  an explicitly selected full game ID. Preflights correctly returned **2**.
- Real model inference, browser/GUI/game episodes, paid judge calls, benchmark
  scores, GPU training and multimodal H/P rounds: **NOT RUN**.

The 2026-11-05 follow-ups completed Vision2Web website/browser/visual scoring
and ARC public-game inference/scorecard; see their runbooks for exact budgets
and limitations. The subsequent [OSWorld Tencent smoke](OSWORLD_TENCENT.md)
completed one desktop task, recording and official evaluation (score 0, model
failure; successful execution is not task success). Multimodal Slime data,
vision tensors/logprobs/masks and domain-specific H-step edit gates remain separate
engineering work, not capabilities supplied by these evaluation adapters.

The existing recursive code hash now includes these Python modules. Historical
SWE runs may fail resume checks after this change; use the original code revision
to resume them. No saved hashes or validation rules were bypassed.
