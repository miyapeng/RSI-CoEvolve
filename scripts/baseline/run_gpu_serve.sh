#!/usr/bin/env bash
# Same server and tool-call check as baseline eval; no task/API/GRPO invocation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${GPU_SERVE_CONFIG:?Set the path to the fixed GPU smoke YAML}"
export PYTHONPATH="$ROOT:$ROOT/mini-swe-agent/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MSWEA_SILENT_STARTUP=1 OMP_NUM_THREADS=2
SERVE_OUTPUT="${SERVE_OUTPUT:-/data/results/rsi-coevolve-gpu-serve-${JOB_ID:?}}"
mkdir -p "$(dirname "$SERVE_OUTPUT")"
status=0
timeout --signal=TERM --kill-after=30s 900s "${GPU_PYTHON:-python}" \
  -m baseline.serving --config "$GPU_SERVE_CONFIG" --output "$SERVE_OUTPUT" || status=$?
if [[ -d "$SERVE_OUTPUT" ]]; then
  cp "$GPU_SERVE_CONFIG" "$SERVE_OUTPUT/config.resolved.yaml"
  ARCHIVE_ROOT="${GPU_ARCHIVE_ROOT:-$ROOT/runs/gpu-smoke-$(date -u +%Y%m%d)}"
  mkdir -p "$ARCHIVE_ROOT"
  cp -a "$SERVE_OUTPUT" "$ARCHIVE_ROOT/job-${JOB_ID:-local}"
fi
exit "$status"
