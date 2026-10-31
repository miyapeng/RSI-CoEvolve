#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}
python_bin=${python_bin:-python}
cd "$project_dir"
export PYTHONNOUSERSITE=1 MSWEA_SILENT_STARTUP=1
"$python_bin" -m pip check
"$python_bin" -m ruff check baseline tests/baseline
"$python_bin" -m compileall -q baseline
"$python_bin" -m pytest tests/baseline -q

# Run unmodified upstream tests. Avoid the editable mini package's tests namespace.
slime_root="$project_dir/../references/slime"
site_dir=$("$python_bin" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cd "$slime_root"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$slime_root:$site_dir" \
  "$python_bin" -S -m pytest tests/test_agent/test_adapters.py tests/test_agent/test_trajectory_manager_branching.py -q
