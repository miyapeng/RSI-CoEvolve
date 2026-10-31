#!/usr/bin/env bash
# Inspect the GPU image as supplied. Do not activate the CPU Conda environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$ROOT/../references/slime}"
export PYTHONPATH="$SLIME_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${MEGATRON_ROOT:-}" ]]; then
  test -f "$MEGATRON_ROOT/megatron/training/arguments.py"
  export PYTHONPATH="$MEGATRON_ROOT:$PYTHONPATH"
fi
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
PROBE_OUTPUT="${PROBE_OUTPUT:-/data/results/rsi-coevolve-gpu-probe-${JOB_ID:?}}"
mkdir -p "$(dirname "$PROBE_OUTPUT")"
status=0
timeout --signal=TERM --kill-after=15s 900s "${GPU_PYTHON:-python}" \
  "$ROOT/scripts/baseline/probe_gpu.py" \
  --model "${MODEL_PATH:-$ROOT/models/Qwen3.5-9B}" --output "$PROBE_OUTPUT" || status=$?
# Keep a small, readable copy on the owner's shared directory, including failures.
if [[ -d "$PROBE_OUTPUT" ]]; then
  ARCHIVE_ROOT="${GPU_ARCHIVE_ROOT:-$ROOT/runs/gpu-smoke-$(date -u +%Y%m%d)}"
  mkdir -p "$ARCHIVE_ROOT"
  cp -a "$PROBE_OUTPUT" "$ARCHIVE_ROOT/job-${JOB_ID:-local}"
fi
exit "$status"
