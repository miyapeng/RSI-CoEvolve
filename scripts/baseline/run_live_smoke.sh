#!/usr/bin/env bash
# Explicit small live run: no GPU, no optimizer, one optimization task.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/baseline/run_live_smoke.sh runs/new-output-directory" >&2
  exit 2
fi
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_dir"
export PYTHONNOUSERSITE=1 MSWEA_SILENT_STARTUP=1
for name in LLM_CENTER_API_KEY E2B_API_KEY; do
  if [[ -z "${!name:-}" ]]; then
    read -r -s -p "$name: " "$name"
    printf '\n'
  fi
  export "$name"
done
export E2B_DOMAIN="${E2B_DOMAIN:-ap-beijing.tencentags.com}"
export E2B_TEMPLATE="${E2B_TEMPLATE:-code-agent-cfs2-new}"
export PROPOSER_MODEL="${PROPOSER_MODEL:-CLAUDE_5puknv}"
export NO_PROXY="${NO_PROXY:-},llm-center.modelbest.co,modelbest.tencentcloudcr.com"
python -m baseline.live_smoke --output "$1" \
  --task-id open-feature_python-sdk_pr329 \
  --policy-model "${SMOKE_POLICY_MODEL:-GPT_15kraa}" --execute
