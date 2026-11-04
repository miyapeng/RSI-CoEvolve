# Tencent AGS / E2B

## Direct creation check (2026-11-04)

`python -m baseline.sandbox_probe --output runs/new-connectivity-check --execute`
creates one temporary sandbox through the existing `fixture_sandbox` helper,
checks a command and a binary file upload/download, then kills the sandbox.
Without `--execute` it only prints the requested image/resources. The API key
is read from the environment or a hidden terminal prompt; it is never saved.
Use the existing `self-hpo-baseline` Conda environment. The probe uses direct
network routing and a digest-qualified image from the existing local manifest.
It does not run the coding agent, call a model, train, or score a benchmark.

Real result: **PASS**, 4.28 seconds, one sandbox, 1 CPU / 1 GiB, 600-second
maximum lifetime, with successful explicit cleanup. Template:
`code-agent-cfs2-new`; domain: `ap-beijing.tencentags.com`. Commands and binary
roundtrip passed. The existing Scale-SWE image has Python but no Node.js, npm,
Chrome/Chromium, Playwright CLI or Claude CLI on its default PATH. It is not a
ready Vision2Web runtime. No new image was built or uploaded in this check.
The report and lifecycle evidence are in the ignored local directory
`runs/tencent-connectivity-20261104/`. Actual runtime-image attestation remains
null; the report records the requested immutable image reference.

This is a new connectivity check; earlier baseline/GPU results below retain
their original dates and scope.

The baseline uses Tencent sandboxes for H-step tasks, P-step tasks/reward, dev, audit and final Verified / Multilingual / Pro V2 evaluation. Qwen serving and Slime optimization stay on the GPU host. `runtime.run_agent` and the original test builders/parsers are retained.

## Configuration

Pinned client: `e2b==2.62.0`. The `x-custom-config` contract comes from the supplied Tencent instructions and the earlier local prototype: task image, enterprise registry, `/mnt/envd -port 49983`, CPU and memory. This uses the normal Sandbox command/files API, not Jupyter/code-interpreter. No SDK monkey patch or backend fallback is used.

Export `E2B_API_KEY` (SDK-format key), `E2B_DOMAIN` and the actual approved `E2B_TEMPLATE`. Proxy mode also reads the variable named by `proxy_env` (default `http_proxy`). Keep credentials in the job/shell environment, never YAML/Git.

`connection: proxy` passes that proxy explicitly. `direct` omits it and calls **class-level** `Sandbox.connect` after creation with `https://49983-<sandbox_id>.<domain>`, rebuilding envd routing. Failed reconnect/setup kills the original sandbox. Cleanup also runs after command/grader exceptions. No thousands-of-sandboxes concurrency settings are enabled.

Each task needs its actual repository image mirrored into Tencent's registry. A generic code-interpreter template does not supply the task repositories. Inventory commands make no remote calls:

```bash
python -m baseline.sandbox_inventory --split optimization_pool --limit 16 --output runs/images-opt.yaml
python -m baseline.sandbox_inventory --split all-training --output configs/tencent_images.local.yaml
python -m baseline.sandbox_inventory --benchmark pro-v2 --output runs/images-pro-v2.yaml
```

After approved mirroring/prewarming, fill `image` and `digest`; add any final-eval mappings to the configured manifest too:

```yaml
images:
  source.registry/swe/task:source-tag:
    image: your-tencent-registry.example/swe/task:fixed-tag
    digest: sha256:<64 lowercase hex characters>
```

Only images selected for the current command must be filled; unused inventory entries may remain null while you prepare a small smoke. There is no inferred registry-prefix replacement or `latest` fallback. Authenticate host `skopeo` through the approved registry login. Before creation, the adapter checks `skopeo inspect --format '{{.Digest}}' docker://<image>` against the configured digest; tag drift stops execution. The checked registry digest is **not an attested running-image digest**: the latter remains null in records. Freeze/protect image tags for experiments. This implementation did not transfer or prewarm images.

## Isolation and grading

Agent creation requests `secure=True` and `allow_internet_access=False`, following the [official E2B API](https://docs.e2b.dev/network/internet-access). Actual Tencent template enforcement still needs live verification; passing a flag is not proof. Trusted graders use fresh sandboxes with internet enabled, matching the prior grading setup.

Common task preparation moves original Git history to a root-only location, creates a clean diff baseline and runs policy commands under UID 65534 with no-new-privileges and a cleared environment. No host mounts, model/sandbox keys, gold patches or private grader inputs are passed to the policy. Grader uploads are root-only. The source-specific scripts, patch application ladders and scoring parsers are unchanged.

Each original final verifier now accepts one optional `environment_factory` argument. The baseline supplies `GraderEnvironment` backed by Tencent. The historical function names retain `orchard`; when the factory is supplied they neither import nor create Orchard environments. CPU callback tests cover all three graders. This is an explicit interface, not a monkey patch or another agent runtime.

Nonzero command exits become observations; SDK/connection failures remain infrastructure exceptions. Shell-side deadlines complement SDK timeouts. Grading timeouts or incomplete logs cannot silently become a negative task reward.

## First live validation

```bash
python -m baseline preflight --output runs/tencent-preflight
# Grader-only: ONE optimization task; no policy GPU required.
python -m baseline oracle --limit 1 --output runs/tencent-oracle --execute
python -m baseline rollout --limit 1 --output runs/tencent-task --execute
```

Then follow [RUNBOOK.md](RUNBOOK.md) for fixed-H training, search and two rounds. Validate template egress restrictions, unprivileged repository/environment access, patch/test results and cleanup on actual images.

Live status updated 2026-10-31: a real API/Tencent synthetic coding repair and a
Scale-SWE optimization-task oracle have run. The working public Tencent domain,
template, inspected mirror/digest and repeatable commands are in
[LIVE_SMOKE.md](LIVE_SMOKE.md). The supplied credentials were held only in process
memory. No GPU is visible; real 9B training remains NOT RUN. CPU SDK fixtures are
still separate from these live results. Passing UID/private-file checks does not
attest the provider's complete network isolation.

The later cluster oracle also passed in the GPU Conda environment (CPU-only job
1427683): empty/invalid/reference scores 0/0/1, 43 reference tests passed.
Langfang uses the **public endpoint directly** with class-level reconnect; its
whitelist proxy failed TLS, and its internal endpoint timed out in the bounded
probe. See [GPU_CONDA.md](GPU_CONDA.md). Distributed workers read credentials
from owner-only files; only their paths are put into Ray runtime configuration.
