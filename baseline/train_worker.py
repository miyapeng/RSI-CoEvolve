"""Run unchanged Slime training in a dedicated Ray instance; never kill other jobs."""

import os
import runpy
import sys
from pathlib import Path

from .credentials import rollout_environment
from .records import ROOT, read
from .training import command


def main():
    request_path = Path(sys.argv[1]).resolve()
    r = read(request_path)
    cmd = command(r)
    os.environ.update(
        BASELINE_TRAIN_REQUEST=str(request_path),
        WANDB_MODE="disabled",
        TENSORBOARD_DIR=str(Path(r["output"]) / "tensorboard"),
        MSWEA_SILENT_STARTUP="1",
    )
    os.environ["PYTHONPATH"] = os.pathsep.join([r["config"]["slime_root"], str(ROOT), os.environ.get("PYTHONPATH", "")])
    sys.path.insert(0, r["config"]["slime_root"])
    import ray

    # Credentials are loaded by each rollout worker from the private file.
    ray.init(address="local", runtime_env={"env_vars": rollout_environment(r["config"])})
    try:
        sys.argv = cmd[1:]
        runpy.run_path(cmd[1], run_name="__main__")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
