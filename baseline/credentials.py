"""Read owner-only credentials locally; never put their values in Ray runtime_env."""

import os
from pathlib import Path

from dotenv import dotenv_values


def load_credentials(names=None):
    filename = os.environ.get("BASELINE_SECRET_ENV_FILE")
    if not filename:
        return
    path = Path(filename)
    if path.stat().st_mode & 0o077:
        raise PermissionError("BASELINE_SECRET_ENV_FILE must be accessible only to its owner")
    for name, value in dotenv_values(path, interpolate=False).items():
        if value is not None and (names is None or name in names):
            os.environ[name] = value


def rollout_environment(config):
    """Only paths and routing settings are serialized into Ray's logs/state."""
    c = config["runtime"]["sandbox"]
    keys = [
        "BASELINE_TRAIN_REQUEST",
        "JOB_ID",
        "BASELINE_SECRET_ENV_FILE",
        "TENSORBOARD_DIR",
        "PYTHONPATH",
        "MSWEA_SILENT_STARTUP",
        "WANDB_MODE",
        "REGISTRY_AUTH_FILE",
        "NO_PROXY",
        "no_proxy",
        c["template_env"],
        c["domain_env"],
    ]
    if c["connection"] == "proxy":
        keys.append(c["proxy_env"])
    if os.environ.get(c["key_env"]) and not os.environ.get("BASELINE_SECRET_ENV_FILE"):
        raise ValueError("Distributed rollout requires a private BASELINE_SECRET_ENV_FILE, not serialized API keys")
    return {key: os.environ[key] for key in keys if key in os.environ}
