"""Native Slime HTTP/token capture with scripted model/sandbox, not real RL."""

import asyncio
import importlib.util
import sys
from types import SimpleNamespace as NS

from test_baseline import candidate

from baseline.config import load_config
from baseline.data import public_task, tasks
from baseline.records import ROOT, read
from baseline.training import prepare


def test_native_adapter_executes_the_real_harness_hooks(monkeypatch, tmp_path):
    slime = ROOT.parent / "references/slime"
    sys.path.insert(0, str(slime))
    from slime.agent.adapters import OpenAIAdapter
    from slime.agent.aiohttp_threaded import run_app_in_thread
    from slime.utils.types import Sample

    from baseline import execution, slime_rollout

    spec = importlib.util.spec_from_file_location("baseline_slime_fixtures", slime / "tests/test_agent/_fakes.py")
    fakes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fakes)
    c = load_config(ROOT / "configs/baseline_9b.yaml")
    row = tasks(c["dataset_split_manifest"], "optimization_pool", 1)[0]
    h = candidate(tmp_path / "harness")
    r = prepare(c["model_checkpoint"], h, [row], c, tmp_path / "train")
    monkeypatch.setenv("BASELINE_TRAIN_REQUEST", str(tmp_path / "train/request.json"))
    actions = []

    class Sandbox:
        def __init__(self, row, *a, **kw):
            self.task, self.sandbox = public_task(row), NS(sandbox_id="fixture-only")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def execute(self, action, **kw):
            actions.append(action["command"])
            return {
                "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if "COMPLETE" in action["command"] else "source",
                "returncode": 0,
                "exception_info": "",
            }

    monkeypatch.setattr(execution, "open_task", Sandbox)
    monkeypatch.setattr(execution, "grade", lambda *a, **kw: {"score": 0.0, "resolved": False, "test_fixture": True})

    async def scenario():
        texts = {
            (601, 602): "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>",
            (
                701,
                702,
            ): "<tool_call><function=bash><parameter=command>echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT</parameter></function></tool_call>",
        }
        async with fakes.FakeSGLangServer([[(-0.1, 601), (-0.2, 602)], [(-0.3, 701), (-0.4, 702)]]) as upstream:
            service = object.__new__(slime_rollout.CaptureService)
            service.adapter = OpenAIAdapter(
                tokenizer=fakes.FakeTokenizer(outputs=texts), sglang_url=upstream.url, fork_threshold_tokens=0
            )
            service.handle = run_app_in_thread(service.adapter.app, host="127.0.0.1", port=0)
            monkeypatch.setattr(slime_rollout, "_service", service)
            try:
                samples = await slime_rollout.generate(
                    NS(),
                    Sample(
                        index=0,
                        group_index=0,
                        metadata={"task_id": row["metadata"]["instance_id"], "split": "optimization_pool"},
                    ),
                    {"max_new_tokens": 32, "temperature": 0.95},
                )
            finally:
                service.handle.stop()
        trained = [
            (tok, lp)
            for s in samples
            for tok, mask, lp in zip(s.tokens[-s.response_length :], s.loss_mask, s.rollout_log_probs, strict=True)
            if mask
        ]
        assert trained == [(601, -0.1), (602, -0.2), (701, -0.3), (702, -0.4)]
        assert all(s.reward == -1.0 for s in samples)
        assert all(s.metadata["harness_revision"] == r["harness"]["revision"] for s in samples)

    asyncio.run(scenario())
    assert actions == ["ls", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
    trace = read(next((tmp_path / "train/rollouts").glob("*/trajectory.json")))
    assert trace["info"]["config"]["agent_type"] == "baseline_candidate.AgentHarness"
    marks = [r["token"] for r in trace["info"]["harness_marks"]]
    assert marks.count("EXECUTE") == 2 and "ACTED" in marks
