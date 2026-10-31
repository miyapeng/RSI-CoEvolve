"""API protocol contract fixtures; live API results live separately under runs/."""

import json

import httpx
import pytest

from baseline.config import load_config
from baseline.proposer import complete, parse_json
from baseline.records import ROOT


@pytest.mark.parametrize("protocol", ["anthropic_messages", "chat_completions"])
def test_proposer_protocol_json_and_no_credentials_in_records(tmp_path, monkeypatch, protocol):
    from baseline import proposer

    settings = {**load_config(ROOT / "configs/baseline_9b.yaml")["proposer"], "protocol": protocol}
    monkeypatch.setenv(settings["key_env"], "private-fixture-credential")
    monkeypatch.setenv(settings["model_env"], "external-strong-model")
    seen = []
    client_class = httpx.Client

    def respond(request):
        payload = json.loads(request.content)
        seen.append(payload)
        assert payload["model"] == "external-strong-model"
        if protocol == "anthropic_messages":
            assert request.url.path == "/v1/messages"
            assert request.headers["x-api-key"] == "private-fixture-credential"
            assert "response_format" not in payload and "temperature" not in payload
            return httpx.Response(
                200, json={"content": [{"type": "text", "text": '{"ok":true}'}], "stop_reason": "end_turn"}
            )
        assert request.url.path == "/v1/chat/completions"
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": '{"ok":true}'}, "finish_reason": "stop"}
                ],
            },
        )

    class FixtureClient(client_class):
        def __init__(self, **kw):
            super().__init__(transport=httpx.MockTransport(respond), **kw)

    monkeypatch.setattr(proposer.httpx, "Client", FixtureClient)
    value = complete(
        settings,
        {
            "messages": [{"role": "user", "content": "fixture"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        },
        tmp_path,
    )
    assert json.loads(value) == {"ok": True} and len(seen) == 1
    assert "private-fixture-credential" not in "".join(p.read_text() for p in tmp_path.iterdir())


def test_truncated_proposer_is_not_accepted_and_has_no_retry(tmp_path, monkeypatch):
    from baseline import proposer

    settings = load_config(ROOT / "configs/baseline_9b.yaml")["proposer"]
    monkeypatch.setenv(settings["key_env"], "private-fixture-credential")
    monkeypatch.setenv(settings["model_env"], "fixture")
    client_class, calls = httpx.Client, []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "{"}], "stop_reason": "max_tokens"})

    class FixtureClient(client_class):
        def __init__(self, **kw):
            super().__init__(transport=httpx.MockTransport(respond), **kw)

    monkeypatch.setattr(proposer.httpx, "Client", FixtureClient)
    with pytest.raises(ValueError, match="Incomplete"):
        complete(settings, {"messages": [], "max_tokens": 1}, tmp_path)
    assert len(calls) == 1


def test_live_smoke_rejects_non_optimization_tasks_before_api_use(tmp_path, monkeypatch):
    from baseline import live_smoke

    def selected_tasks(manifest, split):
        assert split == "optimization_pool"
        return [{"metadata": {"instance_id": "allowed-training-task"}}]

    monkeypatch.setattr(live_smoke, "tasks", selected_tasks)
    config = load_config(ROOT / "configs/baseline_9b.yaml")
    with pytest.raises(ValueError, match="optimization_pool"):
        live_smoke.run(config, tmp_path / "run", policy_model="fixture", task_id="holdout-task")
    assert not (tmp_path / "run").exists()


def test_native_messages_json_fence_is_explicitly_unwrapped():
    assert parse_json('{"ok":true}') == {"ok": True}
    assert parse_json('```json\n{"ok":true}\n```') == {"ok": True}
    with pytest.raises(json.JSONDecodeError):
        parse_json('Some prose before\n```json\n{"ok":true}\n```')
    with pytest.raises(json.JSONDecodeError):
        parse_json('```json\n{"ok":true}\n```\nadditional instructions')


def test_invalid_proposals_keep_parent_and_do_not_abort_h_step(tmp_path, monkeypatch):
    from baseline import search
    from baseline.harness import revision

    config = load_config(ROOT / "configs/baseline_9b.yaml")
    config["h_proposals_per_round"] = 2
    evaluated = []

    def evaluate(*args):
        evaluated.append(args[3])
        return [{"score": 0.0}]

    calls = []

    def propose(*args):
        calls.append(args)
        if len(calls) == 1:
            return parse_json('```json\n{}\n```\nextra prose')
        return []

    monkeypatch.setattr(search, "evaluate", evaluate)
    monkeypatch.setattr(search, "experience", lambda *a: {})
    monkeypatch.setattr(search, "export_runs", lambda *a: None)
    monkeypatch.setattr(search, "propose", propose)
    parent = ROOT / "harnesses/seed"
    result, history = search.h_step(None, "frozen-policy", parent, [], [], config, tmp_path)
    assert result["revision"] == revision(parent)
    assert len(history) == len(calls) == 2 and len(evaluated) == 2
    assert all(not r["accepted"] and r["policy_checkpoint"] == "frozen-policy" for r in history)
    assert history[0]["error_type"] == "JSONDecodeError"
    assert history[1]["rejection"] == "Proposal must be one JSON object"
    assert len((tmp_path / "candidate_history.jsonl").read_text().splitlines()) == 2

    def api_failure(*args):
        raise RuntimeError("Proposer Messages API returned HTTP 401")

    monkeypatch.setattr(search, "propose", api_failure)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        search.h_step(None, "frozen-policy", parent, [], [], config, tmp_path / "api_failure")


def test_proposer_receives_actual_runtime_contract():
    from baseline.search import runtime_contract

    contract = runtime_contract()
    assert "def validate_code" in contract["validator"]
    assert "def step" in contract["step"]
    assert "def execute_actions" in contract["execute_actions"]
    assert "class Submitted" in contract["submission"]
    assert "self.add_messages(*e.messages)" in contract["run"]
    assert '"submission":' in contract["finish_check"]
