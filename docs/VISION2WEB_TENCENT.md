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

Foreground commands run in the sandbox with private stdout/stderr/exit files
and short status polls. A real Claude invocation hit a streaming timeout after
150 seconds; increasing `request_timeout` did not fix it (E2B 2.62 ignores that
argument for streaming commands). `timeout -k 5` enforces `command_timeout` on
the remote process. There are no command retries. The official caller still
receives the original stdout, stderr and exit status; any transport failure ends
the episode and its sandbox is cleaned up. Detached website startup retains
the SDK's background path.

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

## Build inside a disposable Tencent sandbox

When the development container cannot run Docker builds, the existing Tencent
sandbox can itself act as a temporary CPU builder. This helper uses the pinned
upstream Dockerfile unchanged with the maintained
[Kaniko builder](https://github.com/osscontainertools/kaniko) in its own disposable
container, and pushes to the explicitly named TCR destination. It must never be
run by copying the executor into the development host's root filesystem.
Registry credentials are uploaded to `/kaniko/.docker`, outside the build
context and excluded from snapshots. The injected `/mnt` control plane is also
excluded. No inference key is needed for image building.

```bash
conda activate self-hpo-baseline
export BASELINE_SECRET_ENV_FILE=/user/miyapeng/.secrets/rsi-coevolve.env
python scripts/benchmarks/build_vision2web_e2b.py \
  --builder-image 'modelbest.tencentcloudcr.com/code-interpreter/kaniko:v1.29.0-debug@sha256:b2ac9f31876a3176e8d1a8be6d03a455f79bdd026ca41a67aeac0af63cf88321' \
  --image modelbest.tencentcloudcr.com/code-interpreter/vision2web:official-d80a4a3-UNIQUE_TAG \
  --registry-auth-file /user/miyapeng/.secrets/rsi-registry-auth.json \
  --output /tmp/vision2web-build-UNIQUE_RUN
```

The helper requests 4 CPU / 16 GiB, zero GPUs, a 7200-second sandbox TTL and a
5400-second build deadline. It records `build.json`, `build.log`, `build.sh` and
the pushed digest, and attempts sandbox cleanup on either success or
failure. Use local temporary storage when the shared filesystem is full; copy
successful build records into the project when space permits. A build success
is separate from a browser probe and a real benchmark run.

The builder mirror above was uploaded and its registry digest checked. When
mirroring it elsewhere, first download to local disk through the cluster's
public-network proxy, then upload to TCR directly. Streaming GHCR to TCR without
the proxy hit an upload EOF after approximately ten minutes on this host.

```bash
skopeo copy \
  docker://ghcr.io/osscontainertools/kaniko@sha256:b2ac9f31876a3176e8d1a8be6d03a455f79bdd026ca41a67aeac0af63cf88321 \
  dir:/tmp/vision2web-kaniko
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  REGISTRY_AUTH_FILE=/path/to/private/registry-auth.json \
  skopeo copy dir:/tmp/vision2web-kaniko \
  docker://modelbest.tencentcloudcr.com/code-interpreter/kaniko:v1.29.0-debug
```

A nested Docker/VFS attempt reached Chrome installation but exhausted the
roughly 20-GiB sandbox disk at the next layer. Overlay mounts were unsupported.
Kaniko avoids the VFS copies; it does not change the official Dockerfile. Its
debug-image shell and root-user bootstrap only allow the E2B command API to
start the builder; the Ubuntu base supplies the resulting runtime filesystem.
The helper explicitly restores Kaniko's certificate-path environment for E2B
commands. TLS verification remains enabled.

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

The image prerequisite above was resolved on 2026-11-05; see the new validation below.


## Image upload and live validation, 2026-11-05 UTC

The **unmodified official Dockerfile** was built with Kaniko and pushed to:

```text
modelbest.tencentcloudcr.com/code-interpreter/vision2web:official-d80a4a3-20261105@sha256:9a45ff15a7812250ecf20da6d19215974b38197785ade0c02b31363db686c194
```

TCR independently returned that digest. The successful build/push took
844.06 seconds, 4 CPU / 16 GiB, zero GPUs; the builder was destroyed.
The registry manifest contains eight layers, linux/amd64. Failed Docker/VFS,
Kaniko bootstrap and idle-stream attempts are retained under
`runs/vision2web-build-20261105/`; no failed attempt is counted as success.

The first two runtime requests encountered image preparation; the next request
passed in 7.66 seconds and was cleaned up. AGS performs
[on-demand image preheating](https://cloud.tencent.com/document/product/1814/139916).
A new image may return `ResourceUnavailable.ImagePreheating`; wait before a
new attempt instead of treating it as a model failure. The adapter reports this
specific status without dumping arbitrary SDK request details or retrying calls.

The official SandboxManager browser probe passed in **26.23 seconds**, including
real Chrome screenshot, detached HTTP serving, binary file/stdin round trip,
absence of the builder credential files and sandbox cleanup. Runtime versions:
Node 20.20.2, Claude Code 2.1.221, playwright-cli 0.1.23, Chrome 157.0.8103.38.
The official Node 20 recipe was retained despite npm's Claude engine warning;
the actual CLI version command passed. Evidence:
`runs/vision2web-browser-20261105/report.json`.

LLMCenter's current gateway access IDs used for this smoke are `CLAUDE_ruivhq`
(Sonnet 4.5) and `GEMINI_ooyf06` (Gemini 3 Pro Preview). Separate tiny real PNG
requests returned HTTP 200 from both endpoints; those calls only establish image
input/connectivity, not task completion. The ignored `configs/vision2web.local.yaml`
contains these aliases, endpoint URLs, the existing dataset path, the dedicated
Conda interpreter and the pinned image. It contains credential variable names only.

The first task launches exposed two import issues before model execution:
`official_worker.py` went one parent directory too high, and the generic dispatch
imported mini-SWE's action schema. Both are corrected. SWE serving exports still
point to the original functions; their identity regression was retained. The
next attempt reached Claude but its foreground stream timed out after 150 seconds.
That failed task and all its sandbox cleanup records are retained, without scores.
The replacement short-poll transport passed a separate **208.64-second** browser
probe, including a **160-second silent command**, exact file round trips and
cleanup (`runs/vision2web-browser-idle-20261105/report.json`). Its exit marker is
written atomically; CPU tests also verify nonzero exits and real process timeout.

The final single-task attempt (`runs/vision2web-real-20261105-04/`) entered the
unchanged official Claude Code harness and remained at its system-init event;
no website or `start.sh` was produced. Separate bounded CLI diagnostics in that
same sandbox returned:

- `CLAUDE_ruivhq`: upstream HTTP 404, model nonexistent or not accessible.
- `CLAUDE_2vgats` (Opus 4.8): upstream 503, no available model channel; the CLI's
  own retries also encountered the project's aggregate RPM limit (429).

A small Node HTTPS request from that sandbox returned HTTP 200, so basic network,
TLS and bearer authentication worked. This does **not** establish that full CLI
requests are supported. After those diagnostics, the idle task's Claude process
was explicitly terminated to avoid continued retries. Upstream recorded failure
(exit 143, 884.91 seconds), downloaded its workspace and destroyed the sandbox.
The stop reason, CLI diagnostics and all failed attempts are retained; the
adapter has no automatic retry or model fallback.

**Website generation: BLOCKED. Official visual/functional scoring: NOT RUN.**
There are no benchmark scores to report. `webpage/abc` has three prototype
viewports and zero functional cases, so even a future successful run of this
task would not validate functional scoring. GPU training and multimodal RL were
not run. Total model requests/cost are unknown because native CLI retries did
not yield complete usage records; they are not recorded as zero.

CPU baseline suite: **49 passed, 1 skipped**; the final atomic-marker change also
passed all **5** focused Tencent bridge tests. Ruff, formatting, compile checks,
`git diff --check` and the Vision2Web Conda environment's `pip check` passed.
The official tracked checkout remains clean. The portable result record is
[`validation/vision2web_tencent_20261105.json`](validation/vision2web_tencent_20261105.json).

### Resume the live test after model routing is fixed

The image is already in TCR; rebuilding or uploading it again is unnecessary.
Obtain an access ID that supports the actual Claude Code Messages/tool/image
requests, then edit the ignored `configs/vision2web.local.yaml` model and
functional-model aliases. The visual judge route is configured separately.
Keep the same official source and digest. Load credentials without displaying
them and launch one new output directory:

```bash
conda activate self-hpo-baseline
export BASELINE_SECRET_ENV_FILE=/user/miyapeng/.secrets/rsi-coevolve.env
python - <<'PY'
import os
import subprocess
from baseline.credentials import load_credentials

load_credentials(["E2B_API_KEY", "LLM_CENTER_API_KEY"])
os.umask(0o077)
raise SystemExit(subprocess.run([
    "/user/miyapeng/miniconda3/envs/vision2web/bin/python", "-m", "baseline.benchmarks",
    "run", "--config", "configs/vision2web.local.yaml",
    "--task-id", "webpage/abc", "--limit", "1",
    "--output", "runs/vision2web-real-NEW_RUN",
]).returncode)
PY
```

This is a fresh evaluation attempt, not a resume override. Historical code-hash
checks remain intact; do not force H/P runs made before these changes to resume.

## New credential and model-route diagnosis, 2026-11-05 UTC

A newly supplied credential was stored separately in an owner-only file outside
Git. The original credential file was preserved. Authentication to the model
catalog returned HTTP 200, but actual request capability varied by access ID:

| Access ID | Actual observation with the new credential |
| --- | --- |
| `CLAUDE_ruivhq` | Same upstream model-not-found HTTP 404 in official Claude Code. Credential replacement alone did not restore this route. |
| `CLAUDE_5x3ex6` | Simple text completed, but the image/tool test produced no `Read` or `Bash` events and declined the task. Exit code zero was not counted as capability success. |
| `CLAUDE_01szjs` | Native image + tool-schema request passed; official Claude Code then actually read the test PNG, ran `printf VISION2WEB_TOOL_OK` and identified red, in 14.76 seconds. Sandbox cleanup confirmed. |
| `GEMINI_ooyf06` | Tiny-image Chat Completions request returned HTTP 200 and identified red. Response reported `gemini-3.1-pro-preview`, despite the catalog's `gemini-3-pro-preview` label. |

The working policy route's response reported `claude-sonnet-5`. These are gateway
model identities, not verified weight hashes. Preserve both access IDs and
response identities; this smoke is not a reproduction of the paper's exact model
configuration. The old credential was not tested against the working new route,
so success cannot be attributed solely to replacing the credential.

The strict fixture requires real `tool_use` events, a successful shell result
and the correct image observation. Merely receiving text, HTTP 200 or a CLI exit
code of zero is insufficient. Diagnostic evidence is in
`runs/vision2web-new-key-20261105/`; it contains no benchmark scores. CLI-reported
cost is an estimate, not company gateway billing. No benchmark runtime, prompt,
grader, Dockerfile or training algorithm was changed for this diagnosis.

### Successful single-task run

`runs/vision2web-real-20261105-05/summary.json` reports `complete: true` and
`official_evaluation_success`. The unchanged official harness completed:

1. Model reads prototypes/materials and generates HTML, CSS, JavaScript and
   `start.sh`; inference succeeded in **1738.91 seconds**.
2. A separate fresh Tencent sandbox deploys the submitted website; port 3000
   became ready in **1.9 seconds**.
3. Official screenshot-only workflows capture desktop, tablet and mobile;
   all three official VLM scoring files are present and valid. Evaluation took
   **175.91 seconds**, with three workflows completed and zero failed.
4. Both inference and evaluation sandboxes were destroyed. Total episode time
   was **1915.77 seconds** (31.93 minutes), with 2 CPU / 8 GiB per sandbox, zero GPUs.

Official `ResultAnalyzer` prototype scores (range 0–1):

| Desktop | Tablet | Mobile |
| --- | --- | --- |
| 0.285714 | 0.458333 | 0.305556 |

These measure this generated page's visual fidelity; successful pipeline
completion is not a high-quality score or a benchmark-wide success rate.
This task has **zero functional cases**, so functional grading remains
**NOT EXERCISED**. The scalar training reward remains null; no RL was run.
The visual route's exact per-grading-response model ID is not retained by the
upstream scorer, so the earlier probe's reported model must not be mistaken for
verified identities for all three grading requests.

Claude Code reported 82 turns and an estimated inference cost of **USD 9.2907**.
This is the CLI's estimate, not the enterprise gateway bill; actual total billing
and the visual judge's cost are unknown. Raw reported usage is preserved.
The full inference included automatic context compaction and real browser
checks; no prompt or agent output was manually edited to obtain completion.

Evidence: native `inference.json`, `analysis.json`, `evaluation_artifacts.json`,
`native/.../test_results/prototypes/*_{actual.png,scores.json}` and transport
lifecycle records beneath the episode directory. The portable summary with
artifact SHA256 values is
[`validation/vision2web_new_key_20261105.json`](validation/vision2web_new_key_20261105.json).
No implementation code changed in this follow-up, so CPU unit tests were not
repeated; the live execution and official output validation above were run.

The full single-task attempt uses `configs/vision2web.new-key.local.yaml`, policy
and functional-model access ID `CLAUDE_01szjs`, and the existing visual-judge access
ID `GEMINI_ooyf06`. Load only the sandbox credential from the old file and the new
model credential from its separate file, then use a fresh output directory:

```bash
conda activate self-hpo-baseline
export BASELINE_SECRET_ENV_FILE=/user/miyapeng/.secrets/rsi-coevolve.env
python - <<'PY'
import os
import subprocess
from dotenv import dotenv_values
from baseline.credentials import load_credentials

load_credentials(["E2B_API_KEY"])
os.environ["LLM_CENTER_API_KEY"] = dotenv_values(
    "/user/miyapeng/.secrets/vision2web-model.env", interpolate=False
)["LLM_CENTER_API_KEY"]
os.umask(0o077)
raise SystemExit(subprocess.run([
    "/user/miyapeng/miniconda3/envs/vision2web/bin/python", "-m", "baseline.benchmarks",
    "run", "--config", "configs/vision2web.new-key.local.yaml",
    "--task-id", "webpage/abc", "--limit", "1",
    "--output", "runs/vision2web-real-NEW_RUN",
]).returncode)
PY
```
