# Tencent AGS / E2B

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

Live status: **NOT RUN**. The current process lacks E2B key/template/domain and image mappings, and has no visible GPU. An existing `http_proxy` alone is insufficient. CPU tests use explicit SDK fixtures, not real Tencent rewards or policy learning.
