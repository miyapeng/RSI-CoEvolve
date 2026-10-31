"""Drive Self-HPO's experience, parent check, fire gate and paired promotion scripts."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

from .execution import evaluate, mean_score
from .harness import check_diagnostic_leak, diff, load, revision, snapshot
from .records import ROOT, append, read, write


def script(name, work, *args, extra_env=None):
    env = {**os.environ, "BASELINE_WORKDIR": str(Path(work).resolve()), **(extra_env or {})}
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/h_step" / name), *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    return {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def export_runs(work, name, rows):
    """Map baseline records onto the existing Self-HPO diagnostic file layout."""
    work = Path(work)
    for repeat in sorted({r["repetition"] for r in rows}):
        d = work / "runs" / f"{name}_r{repeat + 1}"
        scores = {}
        for r in (r for r in rows if r["repetition"] == repeat):
            if r["score"] is None:
                raise ValueError("Infrastructure failure cannot be promoted/scored as unresolved")
            iid = r["task_id"]
            target = d / "training_source" / iid / f"{iid}.traj.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r["trajectory"], target)
            scores[iid] = {"resolved": r["score"] == 1}
        write(d / "verify_results_merged.json", scores)
    return work / "runs" / f"{name}_r1"


def experience(work, optimization, rows):
    work = Path(work)
    if any(r["split"] != "optimization_pool" for r in optimization + rows):
        raise ValueError("Offline gold diagnosis is optimization_pool only")
    export_runs(work, "parent_seed", rows)
    for r in optimization:
        append(work / "data/gold_patches.jsonl", {"instance_id": r["metadata"]["instance_id"], "patch": r["patch"]})
    result = script("build_experience.py", work)
    write(work / "build-experience.json", result)
    if result["exit_code"]:
        raise RuntimeError("Upstream experience builder failed")
    return {p.name: p.read_text() for p in (work / "experience").iterdir() if p.is_file()}


def propose(config, client, checkpoint, parent, diagnostics, optimization, traces, history, output):
    stats = json.loads(diagnostics["stats.json"])
    prompt = (ROOT / "scripts/h_step/propose_prompt.tmpl").read_text()
    subs = {
        "N_TASKS": len(optimization),
        "SEARCH_DESC": "optimization_pool only",
        "SEED_BAR": mean_score(traces) * 100,
        "COVERAGE_MISS": stats["coverage_miss_pct_of_failed_with_gold_files"],
        "PRECISION": stats["precision_pct_of_failed_with_gold_files"],
        "OWN_TEST_PASS": stats["own_test_passed_pct_of_failed_with_traj"],
        "OWN_TEST_NEVER": stats["own_test_never_pct_of_failed_with_traj"],
        "SUBLOSS_N": stats["submission_loss_n"],
        "SUBLOSS_PCT": stats["submission_loss_pct_of_failed"],
    }
    for key, value in subs.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    prompt = prompt.replace("250", str(config["runtime"]["max_steps"]))
    prompt = prompt.replace(
        "Override step() (or execute_actions); RL training calls only these two.",
        "Override step() (or execute_actions); train and eval execute the identical full mini loop.",
    )
    prompt += """
API integration: you cannot read or write this host. All evidence is supplied below.
Return ONE JSON object: agent_py (complete code), marker, act_marker, purpose, notes.
config.yaml is frozen. The parent retention check still applies. Do not embed task-specific answers.
Use a literal AgentHarness(ProgressTrackingAgent) class with step/execute_actions and _hook_* helpers.
No run/query overrides, decorators, imports except the existing ProgressTrackingAgent, Submitted, re, json.
Use self.messages, self.n_calls, self.repo_dir, self.repo_diff(), self.env.execute(), self.mark(), self.add_messages().
Instance state must use self._hook_* names. No self.model/config access, host I/O, getattr, eval or introspection.
Use only normal dict/list/string operations and bounded loops. Candidate methods may call super().step()
or super().execute_actions(message). Search and training invoke the identical full mini agent.
The gold patches below are OFFLINE diagnosis only; never include their contents or task IDs in agent code.
"""
    # Include a bounded, deterministic success/failure contrast, with gold only from optimization tasks.
    by_id = {r["metadata"]["instance_id"]: r for r in optimization}
    chosen = [r for score in (0, 1) for r in [r for r in traces if r["score"] == score][:2]]
    evidence = {
        "diagnostics": diagnostics,
        "parent_code": (Path(parent) / "agent.py").read_text(),
        "history": history,
        "examples": [
            {
                "task_id": r["task_id"],
                "score": r["score"],
                "trajectory": Path(r["trajectory"]).read_text()[: config["proposer"]["max_trace_chars"]],
                "gold_patch": by_id[r["task_id"]]["patch"],
            }
            for r in chosen
        ],
    }
    request = {
        "messages": [{"role": "user", "content": prompt + "\n" + json.dumps(evidence)}],
        "max_tokens": config["proposer"]["max_tokens"],
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }
    if config["proposer_mode"] == "api":
        p = config["proposer"]
        request["model"] = os.environ[p["model_env"]]
        write(Path(output) / "proposal-input.json", request)
        with OpenAI(base_url=p["base_url"], api_key=os.environ[p["key_env"]], max_retries=0) as proposer:
            response = proposer.chat.completions.create(**request)
    else:
        request["model"] = checkpoint
        write(Path(output) / "proposal-input.json", request)
        response = client.chat.completions.create(**request)
    write(Path(output) / "proposal-output.json", response.model_dump(mode="json"))
    return json.loads(response.choices[0].message.content)


def h_step(client, checkpoint, harness, optimization, dev, config, output):
    output = Path(output)
    work = output / "search"
    snapshot(harness, work / "agents/parent")
    current = "parent"
    source_rows = evaluate(client, optimization, checkpoint, harness, config, output / "h_rollouts")
    diagnostics = experience(output / "diagnosis", optimization, source_rows)
    baseline = evaluate(client, dev, checkpoint, harness, config, output / "dev_parent")
    export_runs(work, "parent_seed", baseline)
    write(work / "experience/frontier.json", [{"name": "parent", "avg": 100 * mean_score(baseline), "kept": True}])
    history = []
    for index in range(config["h_proposals_per_round"]):
        name = f"candidate_{index:03d}"
        parent, cand = work / "agents" / current, work / "agents" / name
        proposal = propose(
            config, client, checkpoint, parent, diagnostics, optimization, source_rows, history, output / name
        )
        snapshot(parent, cand)
        (cand / "parent").write_text(current)
        record = {
            "candidate": name,
            "policy_checkpoint": checkpoint,
            "parent_revision": revision(parent),
            "accepted": False,
            "candidate_revision": None,
        }
        try:
            for field, filename in [
                ("agent_py", "agent.py"),
                ("marker", "MARKER.txt"),
                ("act_marker", "ACT_MARKER.txt"),
                ("purpose", "purpose"),
            ]:
                value = proposal[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Missing {field}")
                (cand / filename).write_text(value)
            if proposal["marker"].strip() == proposal["act_marker"].strip():
                raise ValueError("Wiring and action markers must differ")
            load(cand)
            check_diagnostic_leak(proposal["agent_py"], optimization)
            record.update(
                candidate_revision=revision(cand),
                purpose=proposal["purpose"],
                notes=proposal.get("notes"),
                diff=diff(parent, cand),
            )
            parent_check = script("parent_check.py", work, name)
            record["parent_check"] = parent_check
            if parent_check["exit_code"]:
                raise ValueError("Parent retention check failed")
        except (ValueError, SyntaxError, KeyError) as exc:
            record["rejection"] = str(exc)
            append(output / "candidate_history.jsonl", record)
            history.append(record)
            continue
        # A small REAL execution gate, before dev evaluation. No mock success marker.
        gate_rows = evaluate(client, optimization[:3], checkpoint, cand, config, output / name / "gate_rollouts")
        gate_path = export_runs(work, f"{name}_firegate", gate_rows)
        gate = script("fire_gate.py", work, name, extra_env={"BASELINE_GATE_TRACES": str(gate_path.resolve())})
        record["fire_gate"] = gate
        if gate["exit_code"] == 0:
            rows = evaluate(client, dev, checkpoint, cand, config, output / name / "dev")
            export_runs(work, name, rows)
            selection = script(
                "record_result.py",
                work,
                name,
                extra_env={"BASELINE_PROMOTE_MARGIN": str(config["promote_margin_percent"])},
            )
            if selection["exit_code"]:
                raise RuntimeError("Upstream paired candidate selection failed")
            result = read(work / "experience/last_candidate.json")
            record.update(
                accepted=result["promoted"],
                evaluation=result,
                mean_tokens=sum(r["tokens"] for r in rows) / len(rows)
                if all(r["tokens"] is not None for r in rows)
                else None,
            )
            if record["accepted"]:
                current = name
        else:
            record["rejection"] = "fire_gate_failed"
        append(output / "candidate_history.jsonl", record)
        history.append(record)
    return snapshot(work / "agents" / current, output / "harness_after"), history
