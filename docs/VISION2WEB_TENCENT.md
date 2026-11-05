# Vision2Web on Tencent AGS/E2B

This adapter runs the **unchanged official Vision2Web harness** in Tencent
sandboxes. It is evaluation-only; it does not enable multimodal Slime training.
The existing SWE execution/training path and the default Vision2Web Docker
configuration are unchanged.

## Implemented boundary

`baseline/benchmarks/vision2web.py` still calls the official `DatasetManager`,
`InferenceEngine`, `EvaluationEngine`, and `ResultAnalyzer` at commit
`d80a4a3d0ea7326541ad87314294269a633789e4`.

The official code invokes Docker directly from both its agent adapter and its
verifier. `vision2web_e2b.py` therefore supplies a **worker-local Docker CLI
transport**, rather than replacing the official agent or monkey-patching its
Python classes. Only that subprocess's PATH changes. There is no local Docker
fallback and no support for arbitrary Docker commands.

| Native operation | Tencent operation |
| --- | --- |
| `images -q` | Check the configured digest-pinned remote reference; not a local pull or proof of tool availability |
| `create`, `start` | Create a fresh AGS sandbox from that TCR image, reconnect its envd client, initialize `/workspace` |
| `exec`, `-w`, `--user`, `-e` | E2B command API, preserving argv quoting, cwd, environment, exit code and output |
| `exec -i` | Binary stdin uploaded to a temporary sandbox file and redirected into the command |
| `exec -d` | Background E2B command; the evaluator still starts `start.sh` and checks localhost port 3000 |
| `cp` / binary `tar` | SDK file API and archives; host paths confined to the episode directory; archive traversal, duplicate paths and chained links rejected |
| `stop` | Kill the sandbox; worker-finally cleanup also handles failures and timeouts |

Inference and evaluation use separate sandbox lifetimes, as upstream does.
`workflow.json` remains host-side during generation and is read by the trusted
verifier only. Public prototype **image files** are distinct from the container
image reference. Screenshots are downloaded as binary files, not embedded into
trajectory JSON.

The explicit Tencent config requests 2 CPU / 8 GiB for the first Claude Code
smoke. This overrides upstream's generic 64 GiB Docker request; both requested
and actual configured resource values are recorded. This is a bounded transport
validation configuration, not an assertion of equal resource budgets to the
paper. Native prompts, actions, grader and VS/FS aggregation are unchanged.

Model credentials are forwarded only as the native CLI's command environment.
The E2B control-plane key stays on the host. Transport audit logs contain
operations, sandbox IDs, resources and error types, not argv/prompts/secret
values. Run directories are private and Git-ignored. Host cluster proxies are
removed from the direct worker environment; public model/package endpoints must
be reachable from the selected Tencent template. Private GPU endpoints need an
explicit reachable network route; localhost on the host is not sandbox localhost.

A native `evaluation_result.json` success can coexist with missing workflow
outputs. `evaluation_artifacts.json` reports missing/invalid verifier files
without changing the original analyzer's zero-filled scores. Incomplete evidence
has `OfficialEvaluationIncomplete`, rather than a claimed successful evaluation.
This checks artifact presence/structure, not grader resistance to malicious sites.

## Environment and data

Use the existing Conda environment when available:

```bash
cd /user/miyapeng/rsi_evolve/self-hpo-baseline
conda activate vision2web
python -m pip check
```

For a new installation: `conda env create -f environment-vision2web.yml`.
No GPU dependencies are needed on this orchestration host. The worker imports
the pinned checkout, even if another editable Vision2Web version is installed.

Reuse the downloaded tasks:

```bash
export VISION2WEB_DATA=/user/miyapeng/Visual-Self-Verification/data/vision2web/extracted
python -m baseline.benchmarks list --config configs/benchmarks/vision2web_tencent.yaml --limit 1
```

The selected smoke task is `webpage/abc`; it is final-test data and must not be
used to tune H/P search. No new dataset was downloaded.

## Prepare the required container image

A SWE image can pass command transport tests while lacking Node, Chrome,
`playwright-cli`, and `claude`. It cannot run this benchmark. An ordinary E2B
code-interpreter image is not sufficient either.

The pinned official source provides `docker/Dockerfile`, not a confirmed public
prebuilt TCR image. It installs Node 20, Chrome, Playwright CLI, Claude Code
2.1.221, and the other official tools. Build that **unmodified** Dockerfile on
a machine/container with working Docker or Podman build privileges:

```bash
# Authenticate interactively; never put registry passwords in command arguments.
export REGISTRY_AUTH_FILE=/path/to/private/registry-auth.json
podman login --authfile "$REGISTRY_AUTH_FILE" modelbest.tencentcloudcr.com
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}modelbest.tencentcloudcr.com"
export no_proxy="$NO_PROXY"
CONTAINER_ENGINE=podman scripts/benchmarks/build_vision2web_image.sh \
  modelbest.tencentcloudcr.com/code-interpreter/vision2web:official-d80a4a3 --push
```

Use the script's returned `tag@sha256` reference:

```bash
export VISION2WEB_SANDBOX_IMAGE='modelbest.tencentcloudcr.com/code-interpreter/vision2web:official-d80a4a3@sha256:ACTUAL_REGISTRY_DIGEST'
```

The script checks upstream commit/dirty state and never changes the host Docker
service. Upstream leaves some package/base-image versions floating, so rebuilds
are **not bit-reproducible**; preserve the resulting registry digest and probe's
tool versions. The company `code-agent-cfs2-new` template injects `/mnt/envd`;
this path has been verified with existing task images. Compatibility with the
new official Web image must pass the probe before any model calls.

## Probe and run

The probe reads E2B credentials from `E2B_API_KEY`, or prompts with terminal echo
disabled. It creates one sandbox with a 600-second TTL and always attempts
cleanup. No model calls are made.

```bash
python -m baseline.benchmarks.vision2web_probe \
  --config configs/benchmarks/vision2web_tencent.yaml \
  --output runs/vision2web-browser-probe-001 --execute
```

It checks official SandboxManager command/file/stdin/background-server behavior,
then the actual browser screenshot and tool versions. `--transport-only` skips
browser/tool checks, reduces resources to 1 CPU / 1 GiB, and **does not establish
benchmark readiness**. Output directories must be new.

For a real task, copy the config to an ignored `configs/*.local.yaml`, then set
all three explicit model roles and their endpoints. The official defaults are:

- Generation: Claude Code, native Anthropic-compatible Messages endpoint.
- Functional verifier: Claude Code, native Anthropic-compatible Messages endpoint.
- Visual judge: multimodal OpenAI-compatible Chat Completions endpoint.

Enterprise gateways require their current access IDs. A text-only model or an
OpenAI-only endpoint cannot substitute for all three roles. The supplied LLM key
is not hardcoded or saved by this adapter. For LLMCenter, generation/functional
base URL is the service root and visual base URL ends in `/v1`; confirm the
configured gateway supports both protocols and visual input. Model overrides
are recorded and must be reported as nonstandard judge configuration.

Inject the credential variables named in that local config, then run:

```bash
python -m baseline.benchmarks preflight --config configs/vision2web.local.yaml
python -m baseline.benchmarks run --config configs/vision2web.local.yaml \
  --task-id webpage/abc --limit 1 --output runs/vision2web-real-001
```

Preflight is read-only; `ready_for_live_attempt` does not verify remote tools or
network access. Each episode records source/config/data hashes, native logs,
site files, screenshot references, official analysis, artifact completeness,
and sandbox lifecycle events. `score` and `training_reward` stay null; native
VS/FS remain separate. Failed runs remain available for diagnosis; retries use
a new output directory. Code changes can invalidate historical H/P resume hashes;
no hash check is bypassed.

## Actual validation on this host, 2026-11-04

- CPU baseline tests: **47 passed, 1 skipped**. Unmodified upstream Slime agent
  regressions: **49 passed, 1 skipped**. Ruff, compile checks, shell syntax and
  Conda `pip check` passed. Tests include command exit codes, SDK secret boundary,
  binary transfer, cleanup on failure, archive traversal rejection and missing
  verifier evidence. These fixture tests do not measure benchmark performance.
- Existing `vision2web` Conda environment reused; added `e2b==2.62.0` and its
  dependencies. Repaired its pre-existing pyecharts dependency omissions with
  Jinja2 3.1.6 / PrettyTable 3.17.0. No GPU environment was modified.
- Real Tencent transport through the **official** SandboxManager: **PASS**,
  10.17 seconds on the final probe, one sandbox, 1 CPU / 1 GiB, cleanup confirmed. Command execution,
  binary upload/stdin/download and detached localhost HTTP serving passed.
  Used the already pinned Scale-SWE image solely for transport verification.
  Evidence: `runs/vision2web-e2b-transport-ready-20261104/report.json`.
  An earlier probe passed in 15.31 seconds; a repeat failed at the HTTP check
  before a startup-readiness wait was added. All three sandboxes were cleaned
  up; all attempts are retained in the portable validation record.
- Existing OpenHands image inventory: created/inspected/killed successfully;
  required Web tools were absent on PATH. No model call was made.
- Existing sibling project's recorded PJLab Web image: registry unavailable
  from this host (proxy EOF; direct DNS failure). TCR lookup of
  `code-interpreter/vision2web` returned 404 with the available credentials.
  These checks do not establish that no suitable image exists elsewhere.
- Official Dockerfile build attempted with Podman: **BLOCKED** while applying
  the Ubuntu base layer: `remount / ... permission denied`. Evidence:
  `runs/vision2web-image-build-20261104/build.log`. Root in this dev container
  does not provide the mount capabilities needed by this builder.
- Browser screenshot, generated benchmark website, official VS/FS and multimodal
  model inference: **NOT RUN**. GPU / RL / formal benchmark sweeps: **NOT RUN**.

The remaining prerequisite is a reachable, compatible Web image in TCR, built
on a build-capable machine or supplied from an existing mirror. Once available,
the browser probe and one-task command above exercise the implemented path.
