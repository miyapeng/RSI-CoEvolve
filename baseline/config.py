from pathlib import Path

import yaml

from .records import ROOT
from .tencent import validate_config


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    for key in ("model_checkpoint", "harness_path", "dataset_split_manifest", "slime_root"):
        config[key] = str((ROOT / config[key]).resolve())
    config["training"]["checkpoint_root"] = str((ROOT / config["training"]["checkpoint_root"]).resolve())
    sandbox = config["runtime"]["sandbox"]
    validate_config(sandbox)
    if sandbox.get("image_manifest"):
        manifest = (ROOT / sandbox["image_manifest"]).resolve()
        sandbox["image_manifest"] = str(manifest)
        sandbox["images"] = yaml.safe_load(manifest.read_text())["images"] if manifest.exists() else {}
    if config["proposer_mode"] not in ("api", "self"):
        raise ValueError("proposer_mode must be api or self")
    for k in (
        "n_outer_rounds",
        "h_proposals_per_round",
        "h_task_sample_size",
        "dev_task_sample_size",
        "p_training_steps",
        "p_task_sample_size",
        "eval_repetitions",
    ):
        if config[k] < 1:
            raise ValueError(f"{k} must be positive")
    return config
