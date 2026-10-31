# GPU Conda environment used for the bounded training smoke

The dedicated prefix is `/user/miyapeng/miniconda3/envs/self-hpo-gpu`.
It is separate from both the CPU baseline environment and the user's MiniCPM environments.
The CUDA Python packages were imported from the already synchronized enterprise image
`share-dev/ngc-docker/ptdbench-slime:dev-96bfe26-cb-55f572c7`, not downloaded again.

Actual preparation on 2026-10-31:

1. Create a fresh Python 3.12 Conda prefix (`environment-gpu.yml` describes the bootstrap).
2. Inside the above image, run `python scripts/baseline/import_image_environment.py --prefix <prefix>`.
   This copies image packages, including its editable SGLang source, into the fresh prefix.
   It preserves Conda's packaging tools, excludes the image's Slime/Megatron editables,
   and records original package versions/source locations in `<prefix>/image-import.json`.
3. Install the local projects using the prefix's Python:

   ```bash
   export PYTHONNOUSERSITE=1
   python -m pip install --no-build-isolation -e ./mini-swe-agent -e .
   python -m pip install --no-deps --no-build-isolation \
     -e ../references/slime -e ../references/Megatron-LM
   ```

   On this machine, agent dependencies were installed offline from the prepared
   `runs/train-smoke-20261031/wheels` directory with `--no-index --find-links`.
   Do not replace image Torch/CUDA packages with CPU wheels.
4. Install `skopeo=1.24.1` with Conda. Its packaged `etc/containers/registries.conf`
   uses legacy v1 syntax; inside this image it conflicts with the v2 registry snippets.
   Back up that file **inside this Conda prefix only**, then use:

   ```toml
   unqualified-search-registries = []
   ```

   All baseline images use fully qualified pinned names. No system Docker settings
   were changed, and no shared daemon was restarted.

Source pins: Slime `0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e`;
Megatron `4b4acac9a1d28ea6829c8d4f566d75698a21249d` plus the two patches documented
in `UPSTREAM.md`. Torch is `2.11.0+cu129`, SGLang `0.5.15.post1`,
Transformer Engine `2.16.1`, FlashAttention `2.8.3`, FLA `0.4.2`, Ray `2.58.0`.
The image differs from the upstream Docker recipe; passing operator/serving probes
alone does not establish that GRPO works.

`pip` reports pre-existing image metadata conflicts for SGLang's FA4, TVM-FFI,
DeepGEMM and TileLang dependencies, and OpenAI version requirements. These have
not been hidden or fixed by upgrading the CUDA stack indiscriminately. The
baseline pins OpenAI 2.54.0 and runs FA2/FlashInfer on A100; actual runtime checks
and any resulting limitations are recorded separately.

## Credentials and network routing

Use owner-readable files outside Git:

- `BASELINE_SECRET_ENV_FILE`: dotenv file containing inference/sandbox credentials;
- `REGISTRY_AUTH_FILE`: registry authentication JSON.

`baseline.credentials` loads these locally. Ray runtime environment records get
file paths and nonsecret routing settings, never API key values. Keep these
paths accessible to workers under the same user's authorization.

Langfang probes established that the Tencent registry is reachable directly,
whereas routing its HTTPS request through the whitelist proxy returned EOF.
The sandbox public endpoint also works directly; the whitelist proxy failed TLS,
and the internal endpoint timed out. Use `connection: direct`,
`E2B_DOMAIN=ap-beijing.tencentags.com`, and
`NO_PROXY=modelbest.tencentcloudcr.com,.tencentags.com`. The existing class-level
SDK reconnect rebuilds envd routing after sandbox creation. The entrypoint appends loopback, cluster service
names and the pod's own addresses to NO_PROXY for local Ray/SGLang traffic.

## Run entrypoint

`bash scripts/baseline/run_training_smoke.sh` requires `GPU_CONDA_PREFIX`,
`BASELINE_CONFIG`, `BASELINE_PHASE` (`oracle`, `rollout`, `p-step`, `run`),
and Cybertron's `JOB_ID`. Optional `BASELINE_TASK_ID` restricts oracle/rollout/P-step
to one **optimization_pool** task without changing split membership.
`BASELINE_TIMEOUT` bounds the process (default 3600 seconds).

Logs and artifacts are written live to the user's writable project area under
`runs/train-smoke-20261031/job-<id>/`; a pointer is written to `/data/results`.
`BASELINE_RUN_ROOT` can override the run root. Checkpoints go to the explicit
`training.checkpoint_root`, not the read-only source model directory.

The actual cluster oracle job `1427683` completed empty/invalid/reference patch
checks with scores 0/0/1. The reference passed 43 tests in a trusted grader sandbox;
this is a scorer check, not a model solution or a learning result. Both allocated
sandboxes were deleted successfully.
