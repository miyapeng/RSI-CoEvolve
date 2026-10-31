import pytest

from baseline.config import load_config
from baseline.credentials import load_credentials, rollout_environment
from baseline.records import ROOT


def test_distributed_workers_receive_secret_path_not_values(monkeypatch, tmp_path):
    secret = tmp_path / "credentials.env"
    secret.write_text("E2B_API_KEY=test-fixture-key\nLLM_CENTER_API_KEY=test-fixture-proposer\n")
    secret.chmod(0o600)
    monkeypatch.setenv("BASELINE_SECRET_ENV_FILE", str(secret))
    monkeypatch.delenv("LLM_CENTER_API_KEY", raising=False)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    load_credentials(["E2B_API_KEY"])
    import os

    assert os.environ["E2B_API_KEY"] == "test-fixture-key"
    assert "LLM_CENTER_API_KEY" not in os.environ
    env = rollout_environment(load_config(ROOT / "configs/baseline_9b.yaml"))
    assert env["BASELINE_SECRET_ENV_FILE"] == str(secret)
    assert "E2B_API_KEY" not in env and "LLM_CENTER_API_KEY" not in env
    secret.chmod(0o644)
    with pytest.raises(PermissionError):
        load_credentials()
    monkeypatch.delenv("BASELINE_SECRET_ENV_FILE")
    with pytest.raises(ValueError, match="private"):
        rollout_environment(load_config(ROOT / "configs/baseline_9b.yaml"))
