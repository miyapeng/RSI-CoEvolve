"""CPU contract tests. Scripted transports/rewards here are NOT experiment results."""

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletion

from baseline.config import load_config
from baseline.data import ROLES, configured_tasks, public_task, split, tasks
from baseline.harness import check_diagnostic_leak, load, revision, snapshot, validate_code
from baseline.outer_loop import run
from baseline.records import ROOT, read, write
from baseline.reward import grade, specification
from baseline.runtime import run_agent
from baseline.search import experience, export_gate_runs, export_runs, script
from baseline.training import command, prepare

HOOK = """from minisweagent.run.benchmarks.swebench import ProgressTrackingAgent
class AgentHarness(ProgressTrackingAgent):
    def step(self):
        self.mark("WIRED")
        if self.n_calls == 0:
            self.add_messages({"role": "user", "content": "Inspect relevant source before editing."})
            self.mark("ACTED")
        return super().step()

    def execute_actions(self, message):
        self.mark("EXECUTE")
        return super().execute_actions(message)
"""


def candidate(path):
    snapshot(ROOT / "harnesses/seed", path)
    (path / "agent.py").write_text(HOOK)
    (path / "MARKER.txt").write_text("WIRED")
    (path / "ACT_MARKER.txt").write_text("ACTED")
    return path


def completion(command):
    return ChatCompletion.model_validate(
        {
            "id": "fixture",
            "object": "chat.completion",
            "created": 0,
            "model": "scripted-test-only",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "CPU fixture",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def test_shared_gpu_serving_and_tool_parser(tmp_path):
    from baseline import execution, serving

    assert execution.serve is serving.serve
    assert execution.serve_command is serving.serve_command
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return completion("printf baseline_tool_smoke")

    client = NS(chat=NS(completions=NS(create=create)))
    serving.tool_smoke(client, "/fixed/checkpoint", tmp_path / "response.json")
    assert requests[0]["model"] == "/fixed/checkpoint"
    assert requests[0]["timeout"] == 60
    assert read(tmp_path / "response.json")["choices"][0]["finish_reason"] == "tool_calls"
    invalid = completion("unused")
    invalid.choices[0].message.tool_calls[0].function.arguments = '{"command": 123}'
    client.chat.completions.create = lambda **kw: invalid
    with pytest.raises(RuntimeError, match="Malformed tool-call"):
        serving.tool_smoke(client, "/fixed/checkpoint", tmp_path / "malformed.json")


def test_full_local_dataset_split_and_grader_construction(tmp_path):
    raw = next((ROOT / "data/training/self-hpo").glob("*/raw/train.jsonl"))
    first = split(raw, tmp_path / "a.json", seed=42)
    assert first == split(raw, tmp_path / "b.json", seed=42)
    assert len(first["tasks"]) == 2484
    assert all(v == 0 for v in first["cross_split_overlap"].values())
    pools = [tasks(tmp_path / "a.json", role) for role in ROLES]
    assert [len(p) for p in pools] == [1987, 249, 248]
    for row in sum(pools, []):
        visible = public_task(row)
        assert set(visible) == {
            "task_id",
            "problem_statement",
            "image",
            "cwd",
            "base_commit",
            "split",
            "dataset_source",
        }
        assert "patch" not in visible and "test_patch" not in visible
        spec = specification(row)
        assert spec.FAIL_TO_PASS and spec.image_name == row["metadata"]["image_url"]
        assert spec.eval_script


def test_configured_small_tasks_cannot_cross_splits():
    c = load_config(ROOT / "configs/baseline_9b.yaml")
    opt = tasks(c["dataset_split_manifest"], "optimization_pool", 2)
    ids = [r["metadata"]["instance_id"] for r in reversed(opt)]
    c["task_ids"] = {"optimization_pool": ids}
    assert configured_tasks(c, "optimization_pool", 1) == [opt[1]]
    assert configured_tasks(c, "optimization_pool", 2) == list(reversed(opt))
    with pytest.raises(ValueError, match="outside"):
        configured_tasks(c, "optimization_pool", 3)
    c["task_ids"]["dev_eval"] = ids
    with pytest.raises(ValueError, match="outside"):
        configured_tasks(c, "dev_eval", 1)
    c["task_ids"]["optimization_pool"] = [ids[0], ids[0]]
    with pytest.raises(ValueError, match="unique"):
        configured_tasks(c, "optimization_pool", 1)


def test_candidate_cannot_embed_offline_gold_path_or_answer():
    row = {"metadata": {"instance_id": "fixture-only-id"}, "patch": "--- a/pkg/private_fix.py\n+++ b/pkg/private_fix.py\n@@ -1 +1 @@\n-old\n+answer\n"}
    for code in ("fixture-only-id", row["patch"], 'hint = "git diff -- pkg/private_fix.py > patch.txt"'):
        with pytest.raises(ValueError):
            check_diagnostic_leak(code, [row])
    check_diagnostic_leak('self.env.execute({"command": "git diff --name-only"})', [row])


def test_same_full_agent_hooks_and_frozen_snapshot(tmp_path):
    from minisweagent.agents.default import DefaultAgent

    config = load_config(ROOT / "configs/baseline_9b.yaml")
    c = {**config["runtime"], "sampling_seed": 42}
    row = tasks(config["dataset_split_manifest"], "optimization_pool", 1)[0]
    h = candidate(tmp_path / "candidate")
    cls, _ = load(h)
    assert cls.run is DefaultAgent.run
    seen = []

    def create(**kw):
        seen.append(kw)
        return completion("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")

    client = NS(chat=NS(completions=NS(create=create)))
    sandbox = NS(
        task=public_task(row),
        sandbox=NS(sandbox_id="fixture"),
        execute=lambda *a, **kw: {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nfixture patch", "returncode": 0},
    )
    for phase in ("search", "train", "eval"):
        r = run_agent(client, sandbox, public_task(row), h, "/fixture/policy", c, tmp_path / phase)
        assert r["termination_reason"] == "Submitted" and r["patch"] == "fixture patch"
        marks = read(r["trajectory"])["info"]["harness_marks"]
        assert {m["token"] for m in marks} == {"WIRED", "ACTED", "EXECUTE"}
    assert seen[0] == seen[1] == seen[2]
    assert row["patch"] not in json.dumps(seen)  # The actual local reference fix is never in requests.
    old = revision(h)
    (h / "agent.py").write_text(HOOK + "\n")
    with pytest.raises(ValueError, match="changed"):
        load(h, old)
    for bad in [
        HOOK.replace('self.mark("WIRED")', 'open("/etc/passwd")'),
        HOOK.replace('self.mark("WIRED")', "self.model.client.close()"),
        HOOK.replace("def step(self)", "def run(self)"),
        HOOK.replace('self.mark("WIRED")', "self.env._sandbox.delete()"),
        HOOK.replace('self.mark("WIRED")', "x = self.env.sandbox.client.api_key"),
        HOOK.replace('self.mark("WIRED")', 'x = "{0._sandbox.client.api_key}".format(self.env)'),
    ]:
        with pytest.raises(ValueError):
            validate_code(bad)


def test_upstream_experience_gate_and_accept_reject(tmp_path):
    config = load_config(ROOT / "configs/baseline_9b.yaml")
    rows = tasks(config["dataset_split_manifest"], "optimization_pool", 3)
    c = {**config["runtime"], "sampling_seed": 42}
    h = candidate(tmp_path / "agents/c1")
    snapshot(ROOT / "harnesses/seed", tmp_path / "agents/parent")
    (h / "parent").write_text("parent")
    records = []
    client = NS(chat=NS(completions=NS(create=lambda **k: completion("submit"))))
    for row in rows:
        sandbox = NS(
            task=public_task(row),
            sandbox=NS(sandbox_id="fixture"),
            execute=lambda *a, **k: {
                "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/x b/x",
                "returncode": 0,
            },
        )
        r = run_agent(
            client, sandbox, public_task(row), h, "fixture", c, tmp_path / "traces" / row["metadata"]["instance_id"]
        )
        records.append({**r, "score": 0.0, "repetition": 0})
    diagnostic = experience(tmp_path / "diagnosis", rows, records)
    assert "coverage_gap.json" in diagnostic
    export_runs(tmp_path, "parent_seed", records)
    gated = export_runs(tmp_path, "c1_firegate", records)
    assert script("parent_check.py", tmp_path, "c1")["exit_code"] == 0
    gate = script("fire_gate.py", tmp_path, "c1", extra_env={"BASELINE_GATE_TRACES": str(gated)})
    assert gate["exit_code"] == 0, gate
    # One task must not be passed to the unchanged upstream two-submission gate.
    with pytest.raises(ValueError, match="three"):
        export_gate_runs(tmp_path / "too_small", records[:1])
    repeats = [{**records[0], "repetition": i} for i in range(3)]
    repeated_gate = export_gate_runs(tmp_path / "repeated_gate", repeats)
    assert len(list(repeated_gate.glob("*/*/*.traj.json"))) == 3
    repeated = script("fire_gate.py", tmp_path, "c1", extra_env={"BASELINE_GATE_TRACES": str(repeated_gate)})
    assert repeated["exit_code"] == 0, repeated
    for repeat in (1, 2):
        trace = next((repeated_gate / f"r{repeat}").glob("*/*.traj.json"))
        content = read(trace)
        content["info"]["exit_status"] = "LimitsExceeded"
        write(trace, content)
    rejected = script("fire_gate.py", tmp_path, "c1", extra_env={"BASELINE_GATE_TRACES": str(repeated_gate)})
    assert rejected["exit_code"] == 1 and "only 1/3" in rejected["stdout"]
    write(tmp_path / "experience/frontier.json", [{"name": "parent", "kept": True, "avg": 0.0}])
    export_runs(tmp_path, "c1", [{**r, "score": 1.0} for r in records])
    assert script("record_result.py", tmp_path, "c1")["exit_code"] == 0
    assert read(tmp_path / "experience/last_candidate.json")["promoted"] is True
    snapshot(h, tmp_path / "agents/c2")
    export_runs(tmp_path, "c2", records)
    assert script("record_result.py", tmp_path, "c2")["exit_code"] == 0
    assert not read(tmp_path / "experience/last_candidate.json")["promoted"]
    assert (tmp_path / "experience/current_parent.txt").read_text().strip() == "c1"


def test_submission_rewrite_uses_actual_mini_exit_message(tmp_path):
    """Regression for a live proposal using obsolete Submitted(string) semantics."""
    config = load_config(ROOT / "configs/baseline_9b.yaml")
    row = tasks(config["dataset_split_manifest"], "optimization_pool", 1)[0]
    h = tmp_path / "candidate"
    snapshot(ROOT / "harnesses/seed", h)
    code = '''from minisweagent.run.benchmarks.swebench import ProgressTrackingAgent
from minisweagent.exceptions import Submitted
class AgentHarness(ProgressTrackingAgent):
    def step(self):
        try:
            return super().step()
        except Submitted:
            self.mark("WIRED")
            patch = self.repo_diff()
            self.mark("ACTED")
            raise Submitted({"role": "exit", "content": patch, "extra": {"exit_status": "Submitted", "submission": patch}})
'''
    (h / "agent.py").write_text(code)
    client = NS(chat=NS(completions=NS(create=lambda **kw: completion("submit"))))

    def execute(action, **kw):
        text = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ninvalid fixture patch" if action["command"] == "submit" else "fixture diff"
        return {"output": text, "returncode": 0}

    sandbox = NS(task=public_task(row), sandbox=NS(sandbox_id="fixture"), execute=execute)
    settings = {**config["runtime"], "sampling_seed": 42}
    for phase in ("search", "train", "eval"):
        result = run_agent(client, sandbox, public_task(row), h, "fixture", settings, tmp_path / phase)
        assert result["termination_reason"] == "Submitted" and result["patch"] == "fixture diff"
        assert {m["token"] for m in read(result["trajectory"])["info"]["harness_marks"]} == {"WIRED", "ACTED"}
    (h / "agent.py").write_text(code.replace('Submitted({"role": "exit", "content": patch, "extra": {"exit_status": "Submitted", "submission": patch}})', 'Submitted(patch)'))
    with pytest.raises(AttributeError, match="get"):
        run_agent(client, sandbox, public_task(row), h, "fixture", settings, tmp_path / "obsolete")


def test_training_plan_dense_no_final_benchmark_and_freeze(tmp_path):
    c = load_config(ROOT / "configs/baseline_9b.yaml")
    c["training"]["gradient_accumulation_fusion"] = False
    c["training"]["recompute_loss"] = True
    rows = tasks(c["dataset_split_manifest"], "optimization_pool", 2)
    request = prepare(c["model_checkpoint"], c["harness_path"], rows, c, tmp_path / "p")
    cmd = command(request)
    assert "--save-hf" in cmd and "--advantage-estimator" in cmd
    assert "--no-gradient-accumulation-fusion" in cmd
    assert "--recompute-loss-function" in cmd
    assert cmd[cmd.index("--log-probs-chunk-size") + 1] == "512"
    assert cmd[cmd.index("--sglang-context-length") + 1] == str(c["runtime"]["max_context_tokens"])
    assert not any("moe" in a or "expert" in a or "eval-prompt" in a for a in cmd)
    text = Path(request["train"]).read_text()
    assert '"patch"' not in text and '"test_patch"' not in text
    assert json.loads(text.splitlines()[0])["prompt"] == [{"role": "user", "content": rows[0]["problem_statement"]}]
    c["runtime"]["top_p"] = 0.95
    with pytest.raises(ValueError, match="nucleus support"):
        command(request)
    c["runtime"]["top_p"] = 1.0
    (Path(request["harness"]["path"]) / "config.yaml").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        command(request)


def test_dense_export_preserves_updates_and_only_carries_frozen_auxiliary(tmp_path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from baseline.checkpoint import complete_dense_export

    def checkpoint(path, tensors):
        path.mkdir()
        save_file(tensors, path / "model.safetensors")
        write(
            path / "model.safetensors.index.json",
            {
                "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
                "weight_map": {k: "model.safetensors" for k in tensors},
            },
        )

    old, new, broken = tmp_path / "source", tmp_path / "updated", tmp_path / "incomplete"
    tensors = {
        "model.language_model.layers.0.weight": torch.tensor([1.0]),
        "lm_head.weight": torch.tensor([2.0]),
        "model.visual.weight": torch.tensor([3.0]),
        "mtp.weight": torch.tensor([4.0]),
    }
    checkpoint(old, tensors)
    checkpoint(
        new, {k: v + 10 for k, v in tensors.items() if k.startswith("model.language_model") or k == "lm_head.weight"}
    )
    checkpoint(broken, {"lm_head.weight": tensors["lm_head.weight"]})
    with pytest.raises(ValueError, match="missing language-model"):
        complete_dense_export(old, broken)
    assert complete_dense_export(old, new)["frozen_auxiliary_tensors_copied"] == 2
    index = read(new / "model.safetensors.index.json")
    assert set(index["weight_map"]) == set(tensors)
    with safe_open(new / "model.safetensors", framework="pt") as f:
        assert f.get_tensor("lm_head.weight").item() == 12.0
    with safe_open(new / "frozen-auxiliary.safetensors", framework="pt") as f:
        assert set(f.keys()) == {"model.visual.weight", "mtp.weight"}
        assert f.get_tensor("mtp.weight").item() == 4.0
    assert complete_dense_export(old, new)["frozen_auxiliary_tensors_copied"] == 0


def test_empty_failure_success_grading_fixture(tmp_path):
    c = load_config(ROOT / "configs/baseline_9b.yaml")["runtime"]
    row = tasks(ROOT / "configs/split_manifest.json", "optimization_pool", 1)[0]
    assert grade(row, "", c, tmp_path / "empty")["score"] == 0
    spec = specification(row)

    class Grader:
        def __init__(self, *a, **k):
            self.sandbox = NS(sandbox_id="test_fixture")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def checked(self, *a):
            pass

        def upload(self, *a):
            pass

        def raw(self, command, *a):
            if "eval.sh" in command:
                return {
                    "returncode": 0,
                    "output": ">>>>> Start Test Output\n"
                    + "\n".join(f"PASSED {t}" for t in spec.FAIL_TO_PASS + spec.PASS_TO_PASS)
                    + "\n>>>>> End Test Output",
                }
            return {"returncode": 0, "output": ""}

    result = grade(row, "fixture patch", c, tmp_path / "success", sandbox_factory=Grader)
    assert result["score"] == 1 and result["executed_tests"]

    class FailedPatch(Grader):
        def raw(self, *a):
            return {"returncode": 1, "output": "patch failed"}

    result = grade(row, "fixture patch", c, tmp_path / "failure", sandbox_factory=FailedPatch)
    assert result["score"] == 0 and result["reason"] == "patch_apply_failed"


def test_two_round_state_resume_cross_pairs_and_modes(tmp_path):
    c = load_config(ROOT / "configs/baseline_9b.yaml")
    c.update(h_task_sample_size=2, dev_task_sample_size=2, p_task_sample_size=2)

    class Ops:
        calls = []
        fail_once = True

        @staticmethod
        @contextlib.contextmanager
        def serve(*a):
            yield None

        def search(self, client, policy, h, *args):
            self.calls.append(("h", policy, revision(h)))
            new = candidate(args[-1] / "candidate")
            return snapshot(new, args[-1] / "selected"), [{"accepted": True}]

        def train(self, policy, h, rows, cfg, out):
            self.calls.append(("p", policy, revision(h)))
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("simulated interruption, TEST ONLY")
            return {
                "checkpoint": policy + "-updated",
                "harness_revision": revision(h),
                "reload_verified": True,
                "simulated": True,
                "statistics": {"test_fixture": True},
            }

        def evaluate(self, client, rows, policy, h, *args):
            return [
                {
                    "task_id": r["metadata"]["instance_id"],
                    "policy_checkpoint": policy,
                    "harness_revision": revision(h),
                    "score": 0.0,
                }
                for r in rows
            ]

    op = Ops()
    with pytest.raises(RuntimeError, match="interruption"):
        run(c, tmp_path / "exp", operations=op, test_only=True)
    state = read(tmp_path / "exp/state.json")
    assert state["next_round"] == 0 and state["pair"]["checkpoint"] == c["model_checkpoint"]
    result = run(c, tmp_path / "exp", resume=True, operations=op, test_only=True)
    assert result["next_round"] == 2
    assert [v[1] for v in op.calls if v[0] == "h"] == [c["model_checkpoint"], c["model_checkpoint"] + "-updated"]
    cells = read(tmp_path / "exp/rounds/round_000/eval_pairs.json")
    assert cells["old_new"][0]["harness_revision"] == cells["new_new"][0]["harness_revision"]
    assert cells["old_new"][0]["policy_checkpoint"] != cells["new_new"][0]["policy_checkpoint"]
    with pytest.raises(ValueError, match="production"):
        run(c, tmp_path / "not_allowed", operations=op)
    for mode, h, p in [("h-only", True, False), ("p-only", False, True), ("base", False, False)]:
        op.calls = []
        run({**c, "h_enabled": h, "p_enabled": p}, tmp_path / mode, operations=op, test_only=True)
        assert any(x[0] == "h" for x in op.calls) == h
        assert any(x[0] == "p" for x in op.calls) == p
