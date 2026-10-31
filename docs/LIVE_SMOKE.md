# Small real Tencent/API smoke

Use the existing **Conda `self-hpo-baseline`** environment. No additional install,
model download, image transfer or GPU allocation was needed for these probes.
This is a real task/transport check, not an RL update or a benchmark score.

## Models and routing

The original Self-HPO HPO setting names Claude Opus 4.8. The enterprise inference
catalog queried on 2026-10-31 instead exposed `CLAUDE_5puknv` with the name
`claude-opus-5-5`, and a real native Messages request succeeded. These are gateway
identifiers/names, not an independently attested provider checkpoint. The proposer
is frozen. `baseline/proposer.py` explicitly supports native Messages or Chat
Completions; the default is Messages. Protocol mismatch is an error, not a silent
model fallback. Missing/truncated JSON is also an error. A single complete JSON
code fence is explicitly unwrapped; extra prose is rejected and the raw response
is retained. This fixes an observed native Messages formatting mismatch.

The successful real-task smoke uses `GPT_15kraa` (gateway name `gpt-6.1-terra`) as
the explicitly selected API execution model. The synthetic fixture used
`QWEN_k5gzkh` (`qwen3.8-max`); its later real-task attempt timed out and remains in
the records. The normal baseline's trainable policy remains local Qwen3.5-9B.
Neither API execution model is an updated 9B checkpoint. API connectivity does
not verify 9B serving or GRPO.

The LLMCenter internal host works directly (`proposer.trust_env: false`). The
current shell's external proxy caused a TLS timeout for this host. Tencent's
working domain here is `ap-beijing.tencentags.com`, template `code-agent-cfs2-new`.
The `.internal.tencentags.com` route did not work from this allocation. Keep the
working external proxy in `http_proxy` for the default sandbox proxy mode. These
observations are specific to this machine; no system proxy settings were changed.

## Reused task assets

The existing local R2E-Gym-e2b runtime removes `docker.io/` and prefixes
`modelbest.tencentcloudcr.com/`. That convention guided discovery, not automatic
acceptance of an unverified mirror. A real sandbox confirmed that the following
optimization-task image contains the repository and the expected base commit:

- Task: `open-feature_python-sdk_pr329`, source Scale-SWE within Self-HPO-RL-Data.
- Mirror: `modelbest.tencentcloudcr.com/aweaiteam/scaleswe:open-feature_python-sdk_pr329`.
- Expected base commit: `c3ad697a80ade72fb5cdee147ac5c11c38e6533f` (exists; task setup resets to it).
- Registry digest: `sha256:5bb2476cad1c647e22e267a9caedbb24efd6cf03948357b9f929177a6bf8b950`.
- Workdir: `/workspace/python-sdk`.

The neighbouring project's prepared Scale-SWE file has 2,263 rows. There are 103
task-ID overlaps with this baseline's optimization pool, all with identical image
references. This does not establish that all 103 mirrors are runnable. No tasks
were moved between splits and no neighbouring final-test task was used as smoke.

## Repeat the one-task probe

```bash
cd /user/miyapeng/rsi_evolve/self-hpo-baseline
source /user/miyapeng/miniconda3/etc/profile.d/conda.sh
conda activate self-hpo-baseline
# One time only if the local manifest does not exist; never overwrite custom mappings.
test -e configs/tencent_images.local.yaml || cp configs/tencent_images.example.yaml configs/tencent_images.local.yaml
# Authenticate interactively, or export REGISTRY_AUTH_FILE to a private auth file.
skopeo login --username 100046455542 modelbest.tencentcloudcr.com
bash scripts/baseline/run_live_smoke.sh runs/my-first-real-task
```

The script prompts without echo if the two API keys are absent, uses the explicit
models above, and refuses to overwrite an existing run directory. It does not
save keys. The first run used a memory-only registry auth descriptor as well;
that descriptor is not a reusable on-disk credential. A GPU job needs registry
authentication on every worker, with `REGISTRY_AUTH_FILE` mounted at the same
path if used. E2B access and host registry authentication are separate.

The command runs one native proposer API check followed by a real API-policy
rollout under unchanged `harnesses/seed`, capped at 20 agent steps / 300 agent
seconds. The normal task image/digest check, UID isolation, patch submission and
source-specific grader all run. A test failure remains a genuine score of zero;
an infrastructure error remains an exception/null score. Success here does not
mean a harness candidate was searched or any policy parameters were trained.

To validate only the real grader, without an inference model:

```bash
python -m baseline oracle --task-id open-feature_python-sdk_pr329 \
  --output runs/my-oracle --execute
```

That trusted branch separately checks empty, invalid and reference patches. Gold
patches stay on the grading side and are never passed to the policy.

An even smaller synthetic coding fixture remains available:

```bash
python -m baseline.live_smoke --output runs/my-synthetic-probe \
  --image modelbest.tencentcloudcr.com/all-hands-ai/runtime/oh_v0.59.0_jqx9rkkdtc8zrris_9yzpn5kns890pwg6:latest \
  --policy-model QWEN_k5gzkh --execute
```

Only this clearly labeled connectivity fixture allows a generic unpinned image.
It does not enter the dataset loop. The real Self-HPO task path still requires its
exact task image and verified registry digest. The synthetic fixture repairs one
function using the same actual AgentHarness, then applies the submitted patch in
a separate grader with six private checks. Its tests are never uploaded to the
agent sandbox. All probe sandboxes have a bounded lifetime and cleanup.

## Remaining real-baseline requirements

- A GPU allocation/image compatible with the pinned Slime/SGLang/Qwen3.5-9B stack.
  Existing CPU Conda is ready for APIs/grading, not CUDA training. No GPU job was submitted.
- Explicit mappings/digest checks for the additional optimization/dev images a
  larger H/P run will select. Reuse existing mirrors; do not bulk-download blindly.
- Real 9B serve/tool-call smoke, one GRPO update, checkpoint export/reload, H-step
  candidate search and two H/P rounds remain separate acceptance checks.

Measured outcomes and failures are recorded in [STATUS.md](STATUS.md). Local raw
responses/trajectories are under `runs/live-smoke-20261031/` and excluded from Git.
