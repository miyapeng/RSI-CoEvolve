"""CPU contract checks. Synthetic test outcomes are not benchmark measurements."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from baseline.benchmarks import get_benchmark, require_swe
from baseline.benchmarks.official import file_reference, path_value
from baseline.benchmarks.swe_mm import ImageClient
from baseline.benchmarks.swe_mm_data import DATASET, REVISION, prepare, public_row
from baseline.records import ROOT, read


def row(iid):
    return {
        "instance_id": iid,
        "repo": "markedjs/marked",
        "version": "1",
        "base_commit": "a" * 40,
        "image": "fixture:latest",
        "problem_statement": "Fix the layout.",
        "image_assets": {
            "problem_statement": ["https://example.org/issue.png"],
            "test_patch": [{"path": "private.png", "url": "https://example.org/PRIVATE.png"}],
        },
        "patch": "PRIVATE_GOLD",
        "test_patch": "PRIVATE_TEST",
        "hints_text": "PRIVATE_HINT",
        "eval_script": "PRIVATE_SCRIPT",
        "log_parser": "parse_log_marked",
        "eval_type": "fail_only",
        "FAIL_TO_PASS": ["PRIVATE_TEST_NAME"],
        "PASS_TO_PASS": [],
    }


@pytest.fixture
def configured(tmp_path):
    config = yaml.safe_load((ROOT / "configs/benchmarks/swe_mm.yaml").read_text())
    config["official"]["dataset_dir"] = str(tmp_path / "dataset")
    prepare(
        tmp_path / "dataset",
        {"dev": [row("x__x-1")], "test": [row("y__y-2")]},
        {"dataset": DATASET, "revision": REVISION},
    )
    return config


def test_public_projection_official_splits_and_fingerprints(configured, tmp_path):
    config = configured
    adapter = get_benchmark(config)
    task = adapter.load_tasks(config, "dev_eval", 1)[0]
    assert task.info.split == "dev" and task.info.task_id == "x__x-1"
    assert "PRIVATE" not in json.dumps(adapter.public_task(task))
    assert public_row(row("x__x-1")) == task.public
    for role in ("optimization_pool", "final_test"):
        with pytest.raises(ValueError, match="evaluation-only"):
            adapter.load_tasks(config, role, 1)
    with pytest.raises(NotImplementedError):
        require_swe(config, "training")
    config["official"]["split"] = "test"
    assert adapter.load_tasks(config, "final_test", 1)[0].info.task_id == "y__y-2"
    with pytest.raises(ValueError, match="evaluation-only"):
        adapter.load_tasks(config, "dev_eval", 1)
    # No resplit leakage; duplicate original IDs are refused.
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(tmp_path / "bad", {"dev": [row("same")], "test": [row("same")]}, {})
    private = Path(config["official"]["dataset_dir"]) / "dev/grader.jsonl"
    private.write_text(private.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        adapter.input_fingerprint(task)


def test_image_input_binding_keeps_private_data_and_base64_out_of_records(tmp_path):
    image = tmp_path / "issue.png"
    image.write_bytes(b"CPU fixture image bytes")
    observation = {"kind": "image", "artifact": file_reference(image)}
    calls = []
    client = NS(chat=NS(completions=NS(create=lambda **kw: calls.append(kw))))
    wrapper = ImageClient(client, [observation])
    request = {
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "public issue"}],
        "model": "fixture",
        "tools": [{"type": "function"}],
    }
    before = copy.deepcopy(request)
    wrapper.chat.completions.create(**request)
    assert request == before
    content = calls[0]["messages"][1]["content"]
    assert content[0]["text"] == "public issue"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "base64" not in json.dumps(observation)
    assert "PRIVATE" not in json.dumps(calls)
    assert calls[0]["tools"] == request["tools"]
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        wrapper.chat.completions.create(**request)


def test_plan_records_agent_transport_and_split(configured, tmp_path):
    adapter = get_benchmark(configured)
    repo = path_value(configured["official"]["repo"])
    if not repo.is_dir():
        pytest.skip("Optional pinned SWE-bench checkout absent")
    task = adapter.load_tasks(configured, "dev_eval", 1)[0]
    plan = adapter.plan(task, configured["model"], repo, configured, tmp_path / "out", 42)
    assert plan["task"]["split"] == "dev"
    assert plan["agent_harness"]["revision"] and plan["grader_patch_hash"]
    assert "PRIVATE" not in json.dumps(plan)
    configured["official"]["runtime"]["max_steps"] += 1
    changed = adapter.plan(task, configured["model"], repo, configured, tmp_path / "out2", 42)
    assert plan["harness_revision"] != changed["harness_revision"]


def test_multimodal_client_uses_same_agent_submit_protocol(configured, tmp_path):
    from openai.types.chat import ChatCompletion

    from baseline.runtime import run_agent

    requests = []

    def create(**kw):
        requests.append(kw)
        return ChatCompletion.model_validate(
            {
                "id": "cpu-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "CPU fixture",
                            "tool_calls": [
                                {
                                    "id": "action",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": json.dumps(
                                            {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    image = tmp_path / "image.png"
    image.write_bytes(b"fixture bytes, no multimodal inference")
    client = ImageClient(
        NS(chat=NS(completions=NS(create=create))), [{"kind": "image", "artifact": file_reference(image)}]
    )
    task = {
        "task_id": "x__x-1",
        "problem_statement": "Public issue",
        "dataset_source": DATASET,
        "split": "dev",
        "image": "fixture:latest",
        "cwd": "/testbed",
    }
    sandbox = NS(
        task=task,
        sandbox=NS(sandbox_id="cpu-fixture"),
        execute=lambda *a, **kw: {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nfixture patch", "returncode": 0},
    )
    runtime = {**configured["official"]["runtime"], "sampling_seed": 42}
    result = run_agent(client, sandbox, task, ROOT / "harnesses/seed", "fixture", runtime, tmp_path / "native")
    assert result["patch"] == "fixture patch" and result["termination_reason"] == "Submitted"
    assert requests[0]["messages"][1]["content"][-1]["type"] == "image_url"
    assert "base64" not in (tmp_path / "native/events.jsonl").read_text()
    assert result["tool_calls"] == 1


def test_grader_receives_image_environment_only(monkeypatch, tmp_path):
    import shlex

    from baseline.benchmarks.swe_mm_grading import image_environment, with_environment

    calls = []
    monkeypatch.setenv("HOST_ONLY_CREDENTIAL", "must-never-enter-sandbox")

    def inspect(command, **kwargs):
        calls.append(command)
        return NS(stdout=json.dumps({"config": {"Env": ["NVM_DIR=/usr/local/nvm", "NOTE=literal $(echo secret)"]}}))

    monkeypatch.setattr(subprocess, "run", inspect)
    runtime = {
        "sandbox": {"images": {"source:latest": {"image": "registry.example/mm:v2", "digest": "sha256:" + "a" * 64}}}
    }
    env = image_environment({"image": "source:latest"}, runtime, tmp_path)
    assert calls[0][-1] == "docker://registry.example/mm@sha256:" + "a" * 64
    assert "HOST_ONLY_CREDENTIAL" not in env
    assert shlex.split(with_environment("node --version", env)) == [
        "env",
        "NVM_DIR=/usr/local/nvm",
        "NOTE=literal $(echo secret)",
        "bash",
        "-c",
        "node --version",
    ]
    assert read(tmp_path / "image_environment.json")["environment"] == env


def test_native_grader_transport_and_real_parser_cpu(tmp_path):
    """Pinned native run_instance with a CPU transport, not a fake reward function.

    Separate process keeps SWE-bench v5 imports out of legacy SWE v4 tests.
    """
    repo = path_value("../references/SWE-bench")
    if not repo.is_dir():
        pytest.skip("Optional pinned SWE-bench checkout absent")
    code = r"""
import json,sys
from pathlib import Path
from types import SimpleNamespace as NS
from baseline.benchmarks.swe_mm_grading import load_grader
from swebench.harness.utils import make_test_spec
from swebench.task.repo import asset_path
module=load_grader(sys.argv[1], sys.argv[2])
task_repo=Path(sys.argv[2])/'task-repo'
(task_repo/'tasks').mkdir(parents=True)
asset=asset_path(task_repo,'x__x-1','fixtures/expected.png')
asset.parent.mkdir(parents=True)
asset.write_bytes(b'PRIVATE_TEST_IMAGE_CPU_FIXTURE')
class Transport:
    id='cpu-fixture'
    def __init__(self, fail=False, crash=False):
        self.fail,self.crash,self.closed=fail,crash,False
        self.copied=[]
    def create_container(self,*a): return self
    def start(self): pass
    def exec_run(self,*a,**kw): return NS(exit_code=0,output=b'')
    def copy_file(self,c,src,dst):
        self.copied.append(str(dst))
        if str(dst)=='/eval.sh':
            text=Path(src).read_text()
            assert text.index('cp /image_assets/') < text.index('>>>>> Start Test Output')
    def execute(self,*a):
        if self.crash: raise RuntimeError('fixture transport failure')
        log='>>>>> Start Test Output\n'
        log+= '1) expected test\n' if self.fail else '1 passing\n'
        return log+'>>>>> End Test Output\n>>>>> Test Exit Code: '+('1' if self.fail else '0')+'\n',False,0.01
    def cleanup(self,*a): self.closed=True
row=dict(instance_id='x__x-1',image='fixture:latest',repo='markedjs/marked',version='1',
         FAIL_TO_PASS=['expected test'],PASS_TO_PASS=[],log_parser='parse_log_marked',
         eval_type='fail_only',eval_script='#!/bin/bash\necho ">>>>> Start Test Output"\necho private test',
         image_assets={'test_patch':[{'path':'fixtures/expected.png','url':'https://unused.invalid/private.png'}]})
pred=dict(instance_id='x__x-1',model_name_or_path='fixture',model_patch='CPU fixture diff')
for mode in ('pass','fail','crash'):
    t=Transport(mode=='fail',mode=='crash')
    r=module.run_instance(make_test_spec(row),pred,None,str(Path(sys.argv[2])/mode),timeout=1,transport=t,task_repo=str(task_repo))
    assert t.closed
    assert '/eval.sh' in t.copied and '/tmp/patch.diff' in t.copied
    assert '/image_assets/fixtures__expected.png' in t.copied
    if mode=='crash': assert r is None
    else: assert r[1]['x__x-1']['resolved'] == (mode=='pass')
import baseline.benchmarks.swe_mm_grading as bridge
closed=[]
class BadPatchSandbox:
    def __init__(self,*a,**kw):
        self.sandbox=NS(sandbox_id='fixture')
        self._transport=NS(sandbox=NS(files=NS(write=lambda *a,**kw:None)))
    def __enter__(self): return self
    def __exit__(self,*a): closed.append(True)
    def raw(self,command,**kw):
        import shlex
        assert command.startswith('env bash -c ')
        action=shlex.split(command)[-1]
        return {'returncode':1 if action.startswith(('git apply','patch --batch')) else 0,'output':'invalid fixture patch'}
bridge.TaskSandbox=BadPatchSandbox
rejected=bridge.grade(row,{'task_id':'x__x-1','image':'fixture:latest'},pred,
                     {'sandbox':{},'grading_timeout':1,'image_environment':{}},sys.argv[1],Path(sys.argv[2])/'rejected')
assert rejected['patch_rejected'] and not rejected['resolved'] and not rejected['infra_failure']
assert closed==[True]
print('native grader + explicit transport: passed (CPU fixtures)')
"""
    env = {**os.environ, "PYTHONPATH": str(repo) + os.pathsep + str(ROOT), "PYTHONNOUSERSITE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code, str(repo), str(tmp_path)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed (CPU fixtures)" in result.stdout
    assert read(tmp_path / "official_grader/provenance.json")["patch_sha256"]
