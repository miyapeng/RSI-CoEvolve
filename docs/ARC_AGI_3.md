# ARC-AGI-3: official online inference smoke

Implemented in `self-hpo-baseline`, the RSI-CoEvolve worktree. On 2026-11-05 UTC,
one **real** `ls20-9607627b` episode completed through the official
`BenchmarkingAgent → Swarm → ARC online API → scorecard` path. It used the
enterprise Anthropic Messages route `CLAUDE_01szjs`, eight actions and no GPU.
Elapsed episode time was **162.67 seconds**. Exit: `ACTION_BUDGET`; official
score: **0.0**, levels completed: **0/7**. This verifies execution, not a solution
or full benchmark performance. [Machine-readable evidence](validation/arc3_live_20261105.json).

ARC's game service owns the environment in this configuration. There is no
Docker image to upload to Tencent, and no E2B sandbox is needed. The existing
SWE and Vision2Web Tencent paths are unchanged. The upstream Standard harness
renders game grids as **text** for the model; this run does not validate image
input, GUI control, multimodal GRPO or ARC harness/policy co-training.

## Implementation boundary

- `baseline/benchmarks/arc_agi_3.py` calls the pinned official Swarm and retains
  its actions, prompts, state management, token accounting and score formula.
  It rejects agent-thread error exits rather than treating them as valid scores.
- `official.model_config_path` selects an external profile without editing the
  official checkout. The worker temporarily points the native loader's
  `MODEL_CONFIG_PATH` at that file; native schema validation still applies.
  The profile's SHA256 is frozen in the execution request/harness revision and
  checked before execution. The configured model must match the profile.
- `official.anthropic_base_url` explicitly records the enterprise endpoint.
  Credentials remain environment references. A log filter protects the Swarm's
  scorecard logging; our saved scorecard excludes the API key.
- `configs/benchmarks/arc_agi_3_smoke.yaml` selects one full game ID and an
  explicit `smoke_test: true`. Its eight-action limit uses the upstream
  `MAX_ACTIONS_HARD_CAP` development facility. Undeclared inherited caps are
  rejected. It is not the official full-budget evaluation protocol.
- `configs/benchmarks/arc_agi_3_models.yaml` uses the native
  `anthropic-python/messages/manual_rolling` runtime, a 300-second agent budget,
  2,048 output tokens per request, disabled thinking and a 45-second request
  timeout. Native SDK retries still apply; the agent time budget is checked
  between actions, not a process-level hard deadline. Model seeds are not sent
  to the API; repeat runs are not promised deterministic.
- `scripts/baseline/arc3_public_access.py` explicitly fetches the official
  anonymous key and catalog. It saves credentials owner-only outside this repo,
  reuses an existing key, and starts neither games nor model calls.

The official repository stays clean at
`353d38be471395b5ae47dbb44df95440677397e1` (MIT). No official source, agent loop,
prompt, action parser or scoring function was copied or changed.

## Conda setup

The existing environment is `/user/miyapeng/miniconda3/envs/rsi-arc3`. It has
Python 3.12.16 and the official frozen runtime dependencies, including
`arc-agi==0.9.9`, `arcengine==0.9.3`, `anthropic==0.125.0`, `openai==2.41.1`.
`PYTHONNOUSERSITE=1` is set in its Conda configuration: otherwise unrelated
packages in the account's user site can contaminate dependency checks.
Neither the SWE CPU nor GPU environment was upgraded.

For a new machine only:

```bash
cd /user/miyapeng/rsi_evolve/self-hpo-baseline
conda env create -f environment-arc3.yml
conda activate rsi-arc3
# First fetch the pinned checkout if absent, using the main baseline environment:
# python scripts/baseline/fetch_dependencies.py --ssh --benchmarks arc-agi-3-benchmarking
mkdir -p runs/arc3-setup
uv export --project ../references/arc-agi-3-benchmarking --frozen --no-dev \
  --no-emit-project --output-file runs/arc3-setup/requirements.lock > /dev/null
uv pip install --python "$CONDA_PREFIX/bin/python" --require-hashes \
  -r runs/arc3-setup/requirements.lock
python -m pip check
```

`uv` here exports/installs an existing lock into **Conda**, without creating a
venv or resolving a new upstream dependency set. The worker imports the pinned
source explicitly, so installing the upstream project again is unnecessary.
The installed package inventory is saved per episode in `environment.json`.

## Access and one bounded live run

Official [available-game documentation](https://docs.arcprize.org/available-games)
allows anonymous access to the three example games (`ls20`, `ft09`, `vc33`).
Other games require a registered ARC API key. The catalog returned 25 IDs during
this run; catalog visibility is not permission to play every game anonymously.
The anonymous endpoint is also used by the [official SDK](https://github.com/arcprize/ARC-AGI).

These commands intentionally contact the official service; the final command
also spends model API tokens. Credentials already exist locally in the two
owner-only files shown below. Do not add their contents to Git or shell tracing.

```bash
cd /user/miyapeng/rsi_evolve/self-hpo-baseline
conda activate rsi-arc3
# This host was tested with direct networking. Unset proxies in this shell only.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
python scripts/baseline/arc3_public_access.py \
  --key-file /user/miyapeng/.secrets/arc3-anonymous.env \
  --output runs/arc3-access/games.json
set -a
source /user/miyapeng/.secrets/arc3-anonymous.env
source /user/miyapeng/.secrets/vision2web-model.env
set +a
unset MAX_ACTIONS_HARD_CAP ANTHROPIC_BASE_URL

python -m baseline.benchmarks preflight \
  --config configs/benchmarks/arc_agi_3_smoke.yaml
python -m baseline.benchmarks run \
  --config configs/benchmarks/arc_agi_3_smoke.yaml \
  --python "$CONDA_PREFIX/bin/python" --limit 1 \
  --output runs/arc3-smoke-next
```

Use a fresh output directory each time. Preflight is offline and checks
credentials by presence, not validity. For another machine override `--python`
and the secret file paths. For a registered key, use a separate private `.env`
file rather than replacing the anonymous credential. For future full evaluation,
make a separate recorded config with explicit IDs, intended native model profile
and budgets, and no smoke cap. Do not represent this eight-action run as such an
evaluation. These adapters accept `final_test` only; ARC tasks are not admitted
to the SWE training/search pools.

## Evidence and tests

Live output: `runs/arc3-live-20261105-01/` (ignored by Git).

- `summary.json`: normalized score, timing, model, harness/code hashes and exit.
- `episodes/r0/ls20-9607627b/native/scorecard.json`: official server scorecard,
  closed at completion. Card ID `cd27f217-909b-412c-b7bc-48118d887a90`.
- `native/*.recording.jsonl`: game frames/actions; sibling `recordings/*/`
  contains all eight `step_*.json` model request/response records, state snapshots
  and `run_meta.json`. These two paths reflect upstream's separate recorders.
- 480,747 native reported total tokens (472,792 input, 7,955 output), including
  repeated history across requests. Actual gateway billing is **unknown**.
  Upstream writes cost `0` without pricing; that means no estimate, not free
  inference. The adapter now exposes `cost_usd: null` for profiles without
  pricing and an explicit `step_records` path. Those metadata fields were added
  after this live run and verified by CPU tests; its original records/hash are
  preserved.
- All 25 live artifact files were checked against the two active credential
  values; no credential matches. No remote Git push or training was performed.

Reproduce CPU checks (no model keys needed):

```bash
conda activate self-hpo-baseline
python -m pytest tests/baseline -q
ruff check baseline scripts/baseline tests/baseline

conda activate rsi-arc3
# Test-tool versions from the official lock; runtime versions remain constrained.
uv pip install --python "$CONDA_PREFIX/bin/python" \
  --constraint runs/arc3-setup/requirements.lock \
  pytest==8.4.1 pytest-asyncio==1.1.0 requests-mock==1.12.1
cd ../references/arc-agi-3-benchmarking
python -m pytest tests/unit/test_model_config.py tests/unit/test_runtime_clients.py \
  tests/unit/test_anthropic_runtime.py tests/unit/test_swarm.py -q
```

The four upstream test modules passed **214 tests**. Project regression results
are recorded in the validation JSON; new tests cover profile identity/hash,
environment restoration, smoke-budget isolation, log redaction and private key
storage/reuse, in addition to the existing official-call/error-exit contracts.

Next useful experiment: deliberately budget a longer run on a public example
and inspect its action trace. Full-suite scores, a game win, local Qwen inference,
image observations and H/P training remain **NOT RUN**. Changing these Python
modules changes the baseline's recursive code hash: resume historical SWE runs
with their original code revision, without bypassing hash checks.
