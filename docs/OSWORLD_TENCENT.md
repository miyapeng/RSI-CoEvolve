# OSWorld-Verified on Tencent AGS

This is an independent evaluation path. It does not run H/P search or training.
The existing local Docker/VM configuration remains available. The Tencent path
uses an existing OSWorld desktop template; no local Docker daemon, KVM device,
new image build or image upload is required.

## Environment and template

The supplied template was actually located in **Guangzhou**:

| Field | Value |
| --- | --- |
| API domain | `ap-guangzhou.tencentags.com` |
| SDK `template` | `osworld-2qqhvdmu2a4` |
| Returned tool ID | `sdt-m63gjp45` |
| Reported resources | 4 CPU, 8192 MB |
| Desktop screenshot | 1920 × 1080, Ubuntu |
| Agent protocol | Official screenshot / `pyautogui` PromptAgent |

A first Beijing lookup returned 404. A Guangzhou probe created one instance,
obtained HTTP 200 from `/platform` and `/screenshot`, and killed the instance.
Evidence: `runs/osworld-tencent-probe-guangzhou-20261105/report.json` and
`desktop.png`. The probe took 16.69 seconds. These checks alone are not a task
score. A template ID does not guarantee that its underlying image stays immutable;
record the Tencent environment version and tool updates for future comparisons.

The host uses Conda `rsi-osworld` (Python 3.12) separately from SWE training.
Official evaluators eagerly import OCR/document packages, so this host profile
includes CPU PyTorch. It does not allocate a GPU. The profile omits unused cloud
provider SDKs, alternative model-agent packages and guest-only GUI packages.

```bash
cd /user/miyapeng/rsi_evolve/self-hpo-baseline
python scripts/baseline/fetch_dependencies.py --ssh --benchmarks OSWorld ags-cookbook
conda env create -f environment-osworld.yml
conda activate rsi-osworld
# Install the recorded versions; CPU wheels are deliberate.
uv pip install --python "$CONDA_PREFIX/bin/python" --torch-backend cpu \
  -r requirements/osworld-eval.lock
python -m pip check
```

For a new dependency resolution, `requirements/osworld-eval.in` preserves the
applicable upstream constraints; do not silently update the tested lock. The
SDK version is `e2b==2.62.0`. The host also needs the `patch` executable. Source
pins are checked before running, including the Tencent cookbook and injection
patch hash. Existing checkouts are never reset by the dependency helper.

## Execute

Provide `E2B_API_KEY` and `LLM_CENTER_API_KEY` through private environment variables.
The SDK accepts the `e2b_...` form. Do not put either value in YAML, Git, a command
argument or guest files. The configuration stores only environment-variable names.
The tested route is direct; stale Cybertron HTTP proxies must not intercept
localhost proxies or Tencent connections. Proxy routing on other clusters needs
a separate connectivity check.

```bash
conda activate rsi-osworld
cd /user/miyapeng/rsi_evolve/self-hpo-baseline
# After loading the two credential variables privately:
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1
python -m baseline.benchmarks preflight \
  --config configs/benchmarks/osworld_tencent.yaml
python -m baseline.benchmarks run \
  --config configs/benchmarks/osworld_tencent.yaml --limit 1 \
  --output runs/osworld-tencent-001
```

Preflight makes no model calls or sandbox allocations. The run command does.
The default is one official OS task (enable Do Not Disturb), at most eight model
turns, one sandbox at a time, 1800-second sandbox TTL and 90-second model request
timeout, plus a 60-second screenshot HTTP timeout. Native 60-second setup and 20-second pre-evaluation waits are retained.
`max_steps` counts model turns; a turn may contain multiple actions. This is a
pipeline smoke budget, not the full benchmark protocol. The configured enterprise
model alias is `GEMINI_ooyf06`; request and returned model identities are both
recorded. No model seed is claimed: the inherited protocol does not send one.

Use `--task-id domain/uuid` to select another official task. For a full suite,
make a separate configuration with an explicitly agreed budget and an empty
`official.task_ids`, then set `--limit` appropriately. The pinned `test_all.json`
has 368 tasks; `test_nogdrive.json` has 360. Neither list should be silently
substituted for another paper's subset. Authentication, external-site changes,
downloads and application-specific setup must be verified before claiming full
coverage. Final benchmark tasks are excluded from H/P adaptation.

## Code boundary

- `osworld.py`: existing adapter and official `run_single_example` invocation.
- `osworld_tencent.py`: explicit provider injection and fresh-instance lifecycle.
  It reuses Tencent's official HTTP/WebSocket proxy, CDP bridge and bootstrap
  helpers, with a versioned HTTP streaming fix. It overrides SDK creation, cleanup and reset. Each reset kills the
  old instance and starts a new instance from the template; Tencent's original
  no-op snapshot restore is not used. Failed kill remains an error with an ID
  retained in the lifecycle log.
- `osworld_provider.patch`: three small constructor changes in a per-episode copy
  of the pinned OSWorld package: provider-pair argument, injection and marking
  a newly created Tencent instance clean. All reset/action/setup/evaluate methods
  remain equivalent at the AST level. Upstream source stays clean.
- `osworld_ags_stream.patch`: successful HTTP responses are forwarded in chunks.
  The native proxy buffered an entire 1.7 MB screenshot for 25–28 seconds on this
  link, exceeding OSWorld's 10-second idle read timeout. Streaming preserves bytes and also applies to video/file downloads.
- `osworld_screenshot.patch`: configurable screenshot HTTP idle timeout, 60 seconds
  in the Tencent profile. Some first-byte waits also exceeded the original 10
  seconds, so streaming alone was insufficient. This applies to the read-only
  screenshot call and readiness check; action execution/replay rules, task step
  limits, evaluator and model budget are unchanged. The value is recorded in the
  resolved config and propagated only inside the isolated worker.
- `osworld_recording_guest.patch`: supports the inspected Tencent modular launcher
  in the official transactional recorder installer. Exact launcher/server/HTTP
  module hashes are checked. Only recording handlers are replaced with the pinned
  official handlers; command, observation and task routes stay Tencent's originals.
  Native service discovery, atomic installation, health/build checks and rollback
  remain unchanged. Unknown template layouts fail rather than disabling checks.
- `osworld_gateway.py`: only replaces PromptAgent's model HTTP call to accommodate
  an enterprise access ID. The official system prompt, screenshot history,
  `predict`, action parser and stop protocol remain in the upstream agent.

The older complete Tencent DesktopEnv overlay is intentionally not copied over
the newer OSWorld source: that would discard recording updates, keyboard-repeat
checks and setup/evaluation failure propagation. Recording updates run in native
`auto` mode; keyboard auto-repeat is disabled and verified by native code.
The provider also uses Tencent's official guest CDP bridge, needed for browser
setup/getters; a screenshot alone does not establish CDP functionality.

Agent inputs contain the public instruction and screenshot history. Private
`config`/`evaluator` records are supplied only to host-side setup and verification.
The model HTTP client and its credential stay on the host. Screenshot records
use PNG file references and hashes; base64 is used only for the model request.
The native trajectory, video, result and model responses remain in the local run.

Outputs include `request.json`, `environment.json`, `sandbox_lifecycle.jsonl`,
`model_calls/`, `native_source/`, native task screenshots/trajectory/recording,
`verification.json`, and a normalized result. Failed setup/model/evaluation/cleanup
does not become a zero score or successful result. API usage is reported as
returned; monetary cost is not inferred. Runs and credentials are Git-ignored.

For publication, patch files were normalized to zero-context diffs. Applying
them produces byte-identical source to the live run; original patch files remain
in that run's `source_patches/`, and both sets of hashes are in the measured record.

Source/code changes can invalidate historical run hashes. Use the original code
revision to resume those runs; this adaptation does not bypass hash checks.

## Verification record

Connection diagnostics found two incompatibilities before any model call:
1. Buffered HTTP screenshots exceeded native idle timeouts.
2. The Tencent `/opt/osworld-server/main.py` launcher delegates to `src.server`,
   so the original recorder installer refused its unfamiliar layout.
The first failed probes were cleaned up and retained, with `score: null`.


Measured 2026-11-05: [machine-readable record](validation/osworld_tencent_20261105.json).

- Run: `runs/osworld-tencent-live-20261105-02`, one official task, `simulated=false`.
- **Pipeline completed, official score 0.0.** Gemini opened the notification
  panel, clicked the wrong location, then emitted `DONE`. The official evaluator
  observed `show-banners=true`, correctly rejecting the claimed completion.
  No coordinate correction, evaluator hint or score override was applied.
- Model: requested `GEMINI_ooyf06`, returned `gemini-3.1-pro-preview`; 3 model
  calls and 3 actions (including `DONE`), 348.74 seconds for the episode.
- Gateway-reported totals: 8,346 prompt tokens, 1,985 completion tokens,
  10,331 total. Monetary cost is unknown. No GPU was used.
- Official recorder build verified; native keyboard checks passed. Video:
  3,405,721 bytes, first frame decoded, 1920×1080, 30 fps, 4,559 frames.
  Native recording coverage metadata remains `unknown`; no stronger coverage
  claim is inferred. Video, recorder metadata/logs and screenshots were downloaded.
- Including connectivity/source-layout diagnostics and the failed initial live
  attempt, 9 short-lived sandbox instances were created; all were cleaned up.
  Only the completed episode made model calls (3 total). The Beijing 404 lookup
  allocated no instance.
- Sandbox deletion returned HTTP 204 and the lifecycle log records successful
  cleanup. The first live attempt stopped before model calls on screenshot
  timeout; its failure and cleanup logs remain in `...live-20261105-01`.
- Project regression: **56 passed, 2 skipped** (optional Ray/native model deps).
  All **5 new adapter tests passed** in `rsi-osworld`; **13 Tencent helper tests
  passed**. Selected upstream OSWorld tests: **58 passed, 1 failed** in the clean
  original checkout. That failure is the existing recording boot fixture missing
  `_disable_key_autorepeat`; its assertion/source was not modified. Earlier runs
  outside the proper Conda PATH also exposed missing rsync/Flask/python-xlib;
  those environment issues were resolved for the reported checks.
- Ruff, `git diff --check`, native imports and `pip check` passed. Credentials
  were checked against changed files and are absent. No remote Git push occurred.

Reproduce CPU checks:

```bash
conda activate self-hpo-baseline
python -m pytest tests/baseline -q
python -m ruff check baseline/benchmarks tests/baseline/test_osworld_tencent.py \
  scripts/baseline/fetch_dependencies.py
conda activate rsi-osworld
python -m pytest tests/baseline/test_osworld_tencent.py -q
python -m pytest ../references/ags-cookbook/examples/osworld-ags/tests -q
(cd ../references/OSWorld && python -m pytest \
  tests/test_setup_failure_propagation.py tests/test_recording_update.py \
  tests/test_key_autorepeat.py tests/test_command_no_replay_http.py -q)
```

The last command currently includes the upstream fixture failure described above.
Full-suite evaluation, actual browser/CDP task execution, GUI-HARVEST comparison
and multimodal RL remain **NOT RUN**. Next, validate a small predetermined
cross-application smoke set, including Chrome, before committing full-suite
resources. Do not tune prompts using these final benchmark results. A fair
comparison also requires matching the suite, model, desktop and step budgets.
