# SWE-bench Multimodal v2

`benchmark: swe_mm` means **SWE-bench/SWE-bench_Multimodal**, not SWE-bench
Multilingual. This adapter adds evaluation; it does not add multimodal GRPO,
H-step selection on final tests, or another agent loop.

## Version and data boundary

- Official source: https://github.com/SWE-bench/SWE-bench, MIT,
  `87ab1f6ced28f75ba73ca899dc759b019310944a` (5.0.1).
- Dataset: https://huggingface.co/datasets/SWE-bench/SWE-bench_Multimodal,
  `4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5`.
- Actual pinned Parquet membership: **dev 100 / test 480**, 5 / 11 repositories;
  neither task IDs nor repositories overlap. The dataset card's narrative still
  mentions older counts (102 / 510); the downloaded tables and recorded hashes
  are authoritative for this experiment. The official
  [v2 announcement](https://www.swebench.com/multimodal.html) describes the 480-task
  locally gradable test release. Do not mix scores with older 517-task releases.
- `data/benchmarks/swe-mm/<revision>/{dev,test}/public.jsonl` contains only issue
  text, public issue-image URLs, task/repository/base-commit and container reference.
  `grader.jsonl` retains the original row for the trusted verifier. Gold/test
  patches, hints, expected test names and patch/test image assets are never passed
  to the policy. Raw data and runs are Git-ignored. The existing `data` symlink
  still points to the user's shared assets; it was not moved or recreated.
- Manifest includes original task order/IDs, repository membership, data revision
  and public/private file SHA256s. Dev maps to `dev_eval`; test maps to
  `final_test`. Cross-role use and `optimization_pool` are rejected. This does not
  yet enable SWE-MM inside the H/P training loop.
- Downloaded dataset metadata does not declare a dataset license. The MIT source
  license is not asserted to cover issue screenshots or underlying task projects.
  No benchmark rows or screenshots are vendored into this repository.

## Execution and verification

`baseline/benchmarks/swe_mm.py` delegates to the **existing**
`baseline.runtime.run_agent` and selected mini-SWE-agent harness. The default
remains `harnesses/seed`. `ImageClient` only adds issue screenshots as OpenAI
multipart image inputs to the first user turn; it does not change agent actions,
step/submit logic, prompts or stopping rules. The mini agent's text request log
plus `observations/observations.json` describes the request binding. Image bytes
are encoded only during API transport, never as large base64 JSON artifacts.
This version supports initial issue images, not a new screenshot/view-image tool.
It requires a model/API that supports **both image inputs and bash function calls**.
An OpenAI-compatible endpoint name alone is not evidence of that capability.

Commands use `TaskSandbox` and the existing Tencent transport. A live oracle
exposed that AGS commands do not inherit Docker `ENV`; the adapter now reads
`Config.Env` from the digest-pinned OCI image with host `skopeo`, records it, and
restores it for trusted setup/grading. It never reads host environment values
for this restoration. The JS adapter
adds the task image's Node executable directory to the restricted agent PATH;
model keys and host environment variables are not sent into the agent sandbox.
Submission ends the agent sandbox before any private grading data is read.

The official `make_test_spec` / `run_instance` / `get_eval_report` functions own
patch application, test script execution, test-asset restoration, JavaScript log
parsing, fail-only vs pass-and-fail handling and resolved scores. A small versioned
`swe_mm_transport.patch` adds an explicit transport argument to a **per-episode
copy** of `run_evaluation.py`. Tencent supplies creation, file copy, command
execution/timeouts and cleanup. Original checkout files stay unchanged. Private
visual reference files are staged only in the fresh grader sandbox, using the
upstream restoration ordering after test patches are applied.

Environment failures or missing grader assets yield missing scores/failures,
not fabricated task losses. An officially rejected patch is a model submission
failure (unresolved), not a transport failure; the native final reverse-apply
check distinguishes it without parsing a model-controlled error string. An empty submission uses the official no-prediction
report and needs no grader sandbox. GPU training and the existing SWE v4 package
are unchanged: the worker imports v5 from its pinned source path. Upstream
Docker runs sometimes add `SYS_ADMIN` for browser tests; Tencent equivalence for
all such tasks is **not assumed**. Browser-heavy tasks require their own oracle
validation before a full suite is trustworthy.

Each episode stores source, dataset, runtime/transport patch, agent harness and
image-map hashes, prediction, trajectory, issue artifacts, native logs/report and
normalized result. Existing outer-loop code-hash resume checks are untouched;
these code changes can invalidate resume of historical runs. Never bypass those
checks; keep the earlier checkout when resuming an older experiment.

## Setup and offline checks

Use the existing CPU Conda environment, without upgrading training packages.
On this development shell, bare `python` can still resolve to the Vision2Web
prefix even after activation. The verified interpreter is
`/user/miyapeng/miniconda3/envs/self-hpo-baseline/bin/python`; use that absolute
path (and `--python` for the worker) when necessary. No global shell settings
were changed:

```bash
conda activate self-hpo-baseline
python -m pip install -r requirements/swe-mm-eval.in
python scripts/baseline/fetch_dependencies.py --ssh --benchmarks SWE-bench --check
# On another checkout, omit --check only if the pinned source is missing.

# Already prepared locally. For a NEW output directory on another machine:
python -m baseline.benchmarks.swe_mm_data \
  --output data/benchmarks/swe-mm/4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5
# --parquet-dir /existing/raw/files reuses downloaded tables without network.

python -m baseline.benchmarks list --config configs/benchmarks/swe_mm.yaml --split dev --limit 100
python -m baseline.benchmarks list --config configs/benchmarks/swe_mm.yaml --split test --limit 480
python -m baseline.benchmarks plan --config configs/benchmarks/swe_mm.yaml \
  --split dev --task-id markedjs__marked-1435 --output runs/swe-mm-plan
```

## Tencent images and a bounded dev run

### Complete dev/test setup

The full tables are already local: **100 dev + 480 test** tasks. This is distinct
from having all referenced issue images, verifier assets and executable container
images. Keep both original splits intact; a missing resource never removes a task
from the denominator or changes its split.

The setup now provides two resumable preparation commands. They do not call a
model, create a sandbox, grade test tasks, or train a policy:

```bash
PY=/user/miyapeng/miniconda3/envs/self-hpo-baseline/bin/python
MM=data/benchmarks/swe-mm/4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5

# Fetch all declared files for BOTH splits. Existing successful files are hash-checked.
"$PY" -m baseline.benchmarks.swe_mm_assets --dataset "$MM" --output "$MM/assets" --workers 4

# Small manifests only: resolve each official task's linux/amd64 image and size.
"$PY" -m baseline.benchmarks.swe_mm_images inventory --dataset "$MM" --output "$MM/setup" --workers 2
# No-network summary after an interruption:
"$PY" -m baseline.benchmarks.swe_mm_images inventory --offline --dataset "$MM" --output "$MM/setup"

# Actual container-layer transfer to Tencent; potentially hundreds of GiB.
# Requires registry authentication, NOT the E2B API key. Does not need Docker daemon.
export REGISTRY_AUTH_FILE=/path/to/private/registry-auth.json
"$PY" -m baseline.benchmarks.swe_mm_images mirror --dataset "$MM" --output "$MM/setup" \
  --image-map configs/swe_mm_images.local.yaml --workers 2
```

Preparation returns exit code **2** when resources are incomplete. Re-running
reuses verified files/pins and attempts missing downloads once; there is no
unbounded retry loop. Registry pull-rate errors stop subsequent batches. Use
`skopeo login --authfile "$REGISTRY_AUTH_FILE" docker.io` locally if you have an
authorized Docker Hub account, or arrange an internal mirror. Never put registry
passwords into project YAML. `E2B_API_KEY` authenticates sandbox creation; it
does **not** authenticate Docker Hub or the Tencent container registry. Execution
remains on Tencent E2B throughout. The path is **official Docker Hub image →
Tencent image registry → E2B sandbox**, not another purchased execution service.

Files produced under the Git-ignored dataset directory:

- `assets/public/`: URL-keyed public image files and a SHA256/dimensions manifest.
  A GitHub `/blob/` file-view link is resolved to the corresponding raw file;
  original, requested and redirected URLs remain recorded. Other invalid links
  are explicit failures, never substitute images.
- `assets/grader/`: private verifier files and separate manifest;
  `task_repo/tasks/<id>/test_assets/<repo-relative-path>` matches the native
  official verifier's local-asset convention. Only the trusted verifier uses it.
- `assets/readiness.json`: counts, failed URLs and affected task IDs by split.
- `setup/images/`, `setup/image_inventory.json`: immutable source digests,
  compressed layer sizes, deduplicated byte count, and missing inventory entries.
- `setup/mirror_report.json`: last transfer/verification outcomes. Only observed,
  matching destination digests enter `configs/swe_mm_images.local.yaml`.

Set `official.asset_dir: <dataset_dir>/assets` to require local assets. In this
workspace, **`configs/swe_mm_dev.local.yaml` and `configs/swe_mm_test.local.yaml`**
already select all original tasks, this strict cache, the existing Conda worker,
and the previous Gemini gateway route using environment variable names only.
The older one-task smoke config is preserved. The committed portable config
keeps `asset_dir: null` for the existing on-demand behavior. For another machine,
copy it to a local config, set the cache path and provider route, and explicitly
select `--split dev` or `--split test`.

```bash
# These are offline readiness checks over EVERY selected task, not live evaluation.
"$PY" -m baseline.benchmarks preflight --config configs/swe_mm_dev.local.yaml \
  --limit 100 --output runs/swe-mm-dev-readiness
"$PY" -m baseline.benchmarks preflight --config configs/swe_mm_test.local.yaml \
  --limit 480 --output runs/swe-mm-test-readiness

# Only after readiness + representative browser-task oracle validation:
# These two commands DO incur model and sandbox usage; not executed during setup.
"$PY" -m baseline.benchmarks run --config configs/swe_mm_dev.local.yaml \
  --limit 100 --output runs/swe-mm-dev-full
"$PY" -m baseline.benchmarks run --config configs/swe_mm_test.local.yaml \
  --limit 480 --output runs/swe-mm-test-full
```

Preflight now checks images and cached public/private assets for the requested
`--limit`, rather than checking only the first task. Strict cache runs fail on
missing or changed bytes; they do not fall back to network fetches. Plans include
both asset-manifest hashes, and the actual image files are hash-checked before
use. Grading still delegates unchanged patch application and scoring to the
official verifier with its `task_repo` argument. Code-hash resume protections
remain in force; use the previous checkout for historical runs.

See `docs/validation/swe_mm_setup.json` for the actual resource counts and blockers
at the end of this setup attempt. A completed table download or CPU test does
not imply that all 580 remote tasks have been validated.

Each task has its own official `image` reference. A Vision2Web/OSWorld generic
image cannot replace it. Mirror the **actual** task image from Docker Hub to TCR,
record the source amd64 digest and destination tag/digest, and populate ignored
`configs/swe_mm_images.local.yaml`. Use the existing private registry auth file;
never put its content into YAML, a command line or Git. Example transfer for the
selected dev task (approximately 739 MB compressed):

```bash
skopeo copy \
  docker://docker.io/swebench/sweb.eval.x86_64.markedjs_1776_marked-1435@sha256:d395fcb0a68f27329812cc558047b1595fedadbefef727c8620cbf2e3f2d0f59 \
  dir:/tmp/rsi-swe-mm-marked1435
# Docker Hub uses this cluster's proxy; direct TCR transfer avoids that proxy.
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  REGISTRY_AUTH_FILE=/path/to/private/registry-auth.json \
  skopeo copy dir:/tmp/rsi-swe-mm-marked1435 \
  docker://modelbest.tencentcloudcr.com/code-interpreter/swe-mm-marked-1435:v2
```

The mapping uses the **observed destination digest**, not a guessed digest:

```yaml
images:
  swebench/sweb.eval.x86_64.markedjs_1776_marked-1435:latest:
    image: modelbest.tencentcloudcr.com/code-interpreter/swe-mm-marked-1435:v2
    digest: sha256:<verified-destination-digest>
```

The selected image is already mirrored in this workspace; copy
`configs/benchmarks/swe_mm_images.example.yaml` to the ignored local mapping on
a fresh checkout instead of downloading it again. Source and TCR digests match.

Copy `configs/benchmarks/swe_mm.yaml` to ignored `configs/swe_mm.local.yaml` and
set the real model access ID, API endpoint/key **environment name**, and task ID.
For LLMCenter, use its resolved accessId and `https://llm-center.modelbest.co/v1`.
Export `E2B_API_KEY`, `E2B_DOMAIN=ap-beijing.tencentags.com`,
`E2B_TEMPLATE=code-agent-cfs2-new`, and `REGISTRY_AUTH_FILE` in the host environment.
Do not export provider credentials into the task container.

```bash
python -m baseline.benchmarks preflight --config configs/swe_mm.local.yaml \
  --task-id markedjs__marked-1435 --output runs/swe-mm-preflight
python -m baseline.benchmarks run --config configs/swe_mm.local.yaml \
  --split dev --task-id markedjs__marked-1435 --limit 1 --output runs/swe-mm-dev-001
```

`preflight` does not create a sandbox or make a model call. `run` does both and
runs the verifier. Default budget is one task, 40 agent turns, 900 seconds for
agent execution and 900 seconds for grading. Use a new output directory for
each attempt. For final evaluation, explicitly select `--split test`, a suitably
large `--limit`, and complete image mappings; do not use those results to tune
this adapter or select harness candidates.

## Validation

CPU fixtures verify original dev/test membership, public/private separation,
image input binding and artifact hashes, frozen task/harness records, and native
upstream `run_instance` with pass/fail/transport-failure cases. Fixture outcomes
are not benchmark results. Local baseline regression: **62 passed, 2 skipped**;
selected upstream JavaScript/parser/anti-spoofing/infrastructure tests:
**47 passed** (31 grading/parser checks + 16 task-repository checks). Original
Slime agent tests: **49 passed, 1 skipped**. A real dev screenshot was downloaded and decoded (1299×205 PNG).
A real gateway probe with `GEMINI_ooyf06` (response model
`gemini-3.1-pro-preview`) described that screenshot and returned the expected
`bash` function call. The 256-token probe exhausted its budget; the bounded
2048-token probe passed (1,587 total reported tokens). This proves the image/API/
tool-call boundary, not a coding success or a completed scored episode.

Live sandbox and model validation status is recorded separately in
`docs/validation/swe_mm.json`; no full dev/test suite or multimodal RL has run.


Real Tencent validation on `markedjs__marked-1435` completed:

| Check | Observed result |
| --- | --- |
| Source image / TCR digest | Match: `d395fcb0a68f27329812cc558047b1595fedadbefef727c8620cbf2e3f2d0f59` |
| Empty submission | Official no-prediction report: unresolved; no sandbox needed |
| Invalid patch | Official patch ladder rejects it in a fresh Tencent sandbox: unresolved |
| Reference patch | Official verifier resolves it: **679 specs, 0 failures** |
| Gemini, 16 steps | 16 tool calls, 99,024 reported tokens, 124.17 s; no submission, score 0 |
| Gemini, 40 steps | 40 tool calls, 327,934 reported tokens, 294.69 s; no submission, score 0 |

Both real model runs terminated with `LimitsExceeded`, not infrastructure errors.
They repeatedly explored reproduction scripts and produced no final patch. The
adapter used the official no-prediction result for these runs; there was no
model-generated patch to execute grader tests against. The successful reference
oracle independently validates actual test execution, not model problem-solving.
Agent sandbox deletion returned HTTP 204 in both logs. Grader cleanup calls also
completed. These are one-task plumbing checks, not benchmark success rates.

Local evidence is under `runs/swe-mm-oracle-001` (original environment failure),
`runs/swe-mm-oracle-002` (fixed 0/0/1 oracle), `runs/swe-mm-dev-001`, and
`runs/swe-mm-dev-002`. These ignored directories contain full native traces;
`docs/validation/swe_mm.json` retains a small shareable summary. The real model
route was `GEMINI_ooyf06`, response model `gemini-3.1-pro-preview`, not Qwen3.5-9B.
No GPU, training update, full dev suite or final test task was run. Before bulk
evaluation, mirror and oracle-check the remaining task images, particularly
browser-dependent repositories. Select model/harness settings using dev only.
