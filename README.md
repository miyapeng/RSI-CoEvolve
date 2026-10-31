# RSI-CoEvolve

A fixed-budget Harness–Policy Co-Evolution coding baseline adapted from [Self-HPO](https://github.com/MSR-Orchard/Self-HPO). Default policy: **Qwen3.5-9B**. Default proposer: a frozen external API model.

The execution path is **mini-SWE-agent → Tencent AGS/E2B task sandbox**, with **native Slime GRPO** on the GPU allocation. Model inference/training stays on the GPU host; shell commands, repository edits and trusted tests run in separate remote sandboxes. Search, training and evaluation invoke the same agent implementation.

This repository is an implementation baseline, not a claim of reproduced RL gains. CPU integration tests pass; real Tencent task execution, GPU updates and two live H/P rounds remain **NOT RUN**. See [verification status](docs/STATUS.md).

## Start with existing assets

```bash
git clone git@github.com:miyapeng/RSI-CoEvolve.git
cd RSI-CoEvolve
python scripts/baseline/fetch_dependencies.py --ssh --with-final-eval
conda env create -f environment-baseline.yml
conda activate self-hpo-baseline
# CPU environment only; keep the GPU image's CUDA Torch unchanged.
python -m pip install torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

Dependencies are pinned and placed in the adjacent `../references/` directory. Existing checkouts are verified and never overwritten. No model, dataset or container image is downloaded by that script.

Link an existing asset directory to `data` and the existing Qwen checkpoint to `models/Qwen3.5-9B`. Required training layout:

```text
data/training/self-hpo/321decd5b4bf3fd7fedfd3110d0b12071064af9f/raw/train.jsonl
models/Qwen3.5-9B/{config.json,tokenizer.json,model.safetensors.index.json,...}
```

The frozen manifest contains public task IDs and fingerprints, not gold patches. The 2,484 tasks are split into optimization 1,987 / dev 249 / audit 248. Existing local worktrees already have the asset links; do not recreate them. Weights, raw data, runs, credentials and machine-local settings are Git-ignored.

## Configure Tencent sandboxes

Export `E2B_API_KEY`, `E2B_DOMAIN`, `E2B_TEMPLATE` in your shell/job environment. Proxy mode reads `http_proxy`; dedicated-link mode uses `connection: direct`. Fill exact source-image → mirrored Tencent registry tag/digest mappings in ignored `configs/tencent_images.local.yaml`:

```bash
python -m baseline.sandbox_inventory --split all-training --output configs/tencent_images.local.yaml
# Fill image/digest entries after mirroring/prewarming; inventory does not transfer images.
python -m baseline preflight --output runs/preflight
```

A generic code-interpreter template cannot replace the task's repository image. Registry digests are checked with host `skopeo` before creation. Agent sandboxes request disabled internet, run unprivileged commands and never receive model/provider credentials. Actual provider enforcement still requires live validation. [Tencent setup and limitations](docs/TENCENT_SANDBOX.md).

## Run

```bash
bash scripts/baseline/test_cpu.sh
python -m baseline rollout --limit 1 --output runs/task-001 --execute
python -m baseline p-step --limit 16 --output runs/p-001 --execute
python -m baseline h-step --limit 8 --output runs/h-001 --execute
python -m baseline run --mode hpo --output runs/hpo-001 --execute
python -m baseline resume --mode hpo --output runs/hpo-001 --execute
```

Without `--execute`, execution commands save plans. Live commands require the real GPU stack, sandbox images and credentials; they fail explicitly when prerequisites are absent. `--mode base`, `h-only`, `p-only`, `hpo` use the same outer loop. Four-cell dev evaluation is configurable. Verified / Multilingual / **Pro V2** have a separate final-evaluation entrypoint and never select H/P candidates.

- [Runbook: GPU setup, all commands, cross-eval and final evaluation](docs/RUNBOOK.md)
- [Architecture and information boundaries](docs/ARCHITECTURE.md)
- [Measured status and remaining blockers](docs/STATUS.md)
- [Source pins, modifications and licenses](docs/UPSTREAM.md)

Local development worktree: `self-hpo-baseline`, branch `research/baseline-9b`. It keeps Self-HPO as remote `origin` and publishes to remote `rsi`, using `git push rsi HEAD:main`. A fresh clone of this GitHub repository uses normal `origin/main` and `git push`.

## Attribution

Original Self-HPO source history and licenses are retained. The original 35B/B200 instructions and reported paper results are preserved in [the upstream README](docs/upstream/Self-HPO_README.md); they are not results of this 9B adaptation. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). No research manuscript has been rewritten.
