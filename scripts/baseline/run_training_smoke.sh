#!/usr/bin/env bash
# A bounded real entrypoint; credentials remain in the owner's private files.
set -euo pipefail
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${BASELINE_CONFIG:?Set an explicit fixed experiment YAML}"
: "${BASELINE_PHASE:?Use oracle, rollout, h-step, p-step, run, or resume}"
: "${GPU_CONDA_PREFIX:?Set the prepared GPU Conda prefix}"
case "$BASELINE_PHASE" in oracle|rollout|h-step|p-step|run|resume) ;; *) exit 2 ;; esac
export PATH="$GPU_CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1 MSWEA_SILENT_STARTUP=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/../references/Megatron-LM:$ROOT/../references/slime:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=2
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,0.0.0.0,.svc,.cluster.local,$(hostname -I | tr ' ' ',')"
export no_proxy="$NO_PROXY"
export RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false
ARCHIVE="${BASELINE_RUN_ROOT:-$ROOT/runs/train-smoke-20261031}/job-${JOB_ID:?}"
mkdir -p "$ARCHIVE"
OUT="${BASELINE_OUTPUT:-$ARCHIVE/artifacts}"
mkdir -p /data/results
printf '%s\n' "$ARCHIVE" > "/data/results/rsi-coevolve-$JOB_ID.txt"
cp "$BASELINE_CONFIG" "$ARCHIVE/config.resolved.yaml"
cd "$ROOT"
args=(--config "$BASELINE_CONFIG" --output "$OUT" --execute)
if [[ -n "${BASELINE_TASK_ID:-}" ]]; then args+=(--task-id "$BASELINE_TASK_ID"); fi
status=0
timeout --signal=TERM --kill-after=30s "${BASELINE_TIMEOUT:-3600}s" \
  python -m baseline "$BASELINE_PHASE" "${args[@]}" > "$ARCHIVE/launcher.log" 2>&1 || status=$?
printf '{"phase":"%s","job_id":"%s","exit_code":%d}\n' "$BASELINE_PHASE" "$JOB_ID" "$status"
exit "$status"
