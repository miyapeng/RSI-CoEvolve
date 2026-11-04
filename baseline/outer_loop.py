"""Fixed H/P alternation and atomic accepted-pair records. No research trigger or pruning."""

from pathlib import Path

import yaml

from . import execution, search, training
from .audit import environment
from .benchmarks import configured_tasks
from .checkpoint import inspect
from .harness import revision, snapshot
from .records import ROOT, digest, read, sha, write


def core_hash():
    import minisweagent

    files = sorted((ROOT / "baseline").rglob("*.py")) + sorted((ROOT / "scripts/h_step").glob("*.py"))
    files += [ROOT / "scripts/h_step/propose_prompt.tmpl"]
    mini = Path(minisweagent.__file__).parent
    return digest(
        {
            **{str(p.relative_to(ROOT)): sha(p) for p in files},
            **{"mini/" + str(p.relative_to(mini)): sha(p) for p in mini.rglob("*.py")},
        }
    )


def check_pair(pair):
    if revision(pair["harness"]["path"]) != pair["harness"]["revision"]:
        raise ValueError("Accepted harness snapshot changed")
    if pair.get("checkpoint_inspection") != inspect(pair["checkpoint"]):
        raise ValueError("Accepted checkpoint files changed since publication")


def initialize(config, output, *, test_only=False):
    output = Path(output)
    if (output / "state.json").exists():
        raise ValueError("Experiment exists: use resume, or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    h = snapshot(config["harness_path"], output / "initial_harness")
    state = {
        "next_round": 0,
        "pair": {
            "checkpoint": config["model_checkpoint"],
            "harness": h,
            "checkpoint_inspection": None if test_only else inspect(config["model_checkpoint"]),
        },
        "config_hash": digest(config),
        "core_hash": core_hash(),
        "test_only": test_only,
        "manifest_sha256": sha(config["dataset_split_manifest"]),
    }
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write(output / "splits.json", read(config["dataset_split_manifest"]))
    write(output / "environment.json", {"test_fixture": True} if test_only else environment(config))
    write(output / "state.json", state)
    return state


def check_state(config, output, *, test_only=False):
    state = read(Path(output) / "state.json")
    if (
        state["config_hash"] != digest(config)
        or state["core_hash"] != core_hash()
        or state["manifest_sha256"] != sha(config["dataset_split_manifest"])
        or state["test_only"] != test_only
    ):
        raise ValueError("Resume requires identical config/code/data and execution mode")
    if revision(state["pair"]["harness"]["path"]) != state["pair"]["harness"]["revision"]:
        raise ValueError("Accepted harness snapshot changed")
    if not test_only:
        check_pair(state["pair"])
    return state


def attempt(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Failed attempts remain intact; no deletion or silent retry within a call.
    path = directory / f"attempt_{len(list(directory.glob('attempt_*'))):03d}"
    path.mkdir()
    return path


class Operations:
    serve = staticmethod(execution.serve)
    evaluate = staticmethod(execution.evaluate)
    search = staticmethod(search.h_step)
    train = staticmethod(training.train)


def run(config, output, *, resume=False, operations=None, test_only=False):
    if operations is not None and not test_only:
        raise ValueError("Injected test operations cannot produce production accepted pairs")
    op = operations or Operations()
    output = Path(output).resolve()
    state = (
        check_state(config, output, test_only=test_only) if resume else initialize(config, output, test_only=test_only)
    )
    optimization = configured_tasks(config, "optimization_pool", config["h_task_sample_size"])
    train_rows = configured_tasks(config, "optimization_pool", config["p_task_sample_size"])
    dev = configured_tasks(config, "dev_eval", config["dev_task_sample_size"])
    # audit_holdout is deliberately never loaded by this loop.
    while state["next_round"] < config["n_outer_rounds"]:
        t = state["next_round"]
        rd = output / "rounds" / f"round_{t:03d}"
        rd.mkdir(parents=True, exist_ok=True)
        before = state["pair"]
        write(rd / "policy_before.json", {"checkpoint": before["checkpoint"]})
        snapshot(before["harness"]["path"], rd / "harness_before")
        # Recover the narrow crash window after completed-round save and before pair publication.
        if (rd / "completed.json").exists():
            completed = read(rd / "completed.json")
        else:
            try:
                if not (rd / "h_result.json").exists():
                    if config["h_enabled"]:
                        with op.serve(before["checkpoint"], config, attempt(rd / "serve_h")) as client:
                            h, history = op.search(
                                client,
                                before["checkpoint"],
                                before["harness"]["path"],
                                optimization,
                                dev,
                                config,
                                attempt(rd / "h_step"),
                            )
                    else:
                        h, history = before["harness"], []
                    write(
                        rd / "h_result.json", {"harness": snapshot(h["path"], rd / "harness_after"), "history": history}
                    )
                h_result = read(rd / "h_result.json")
                h = h_result["harness"]
                if revision(h["path"]) != h["revision"]:
                    raise ValueError("Selected harness changed before training")
                if not (rd / "old_eval.json").exists():
                    with op.serve(before["checkpoint"], config, attempt(rd / "serve_old_eval")) as client:
                        old_target = op.evaluate(
                            client, dev, before["checkpoint"], h["path"], config, attempt(rd / "old_new")
                        )
                        old_old = (
                            op.evaluate(
                                client,
                                dev,
                                before["checkpoint"],
                                before["harness"]["path"],
                                config,
                                attempt(rd / "old_old"),
                            )
                            if config["cross_eval"]
                            else None
                        )
                    write(rd / "old_eval.json", {"old_old": old_old, "old_new": old_target})
                if not (rd / "p_result.json").exists():
                    if config["p_enabled"]:
                        completed_p = sorted((rd / "p_train").glob("attempt_*/result.json"))
                        if completed_p:
                            # Recover a completed export/reload if the process stopped before p_result publication.
                            if len(completed_p) != 1:
                                raise ValueError("Multiple completed P-steps require explicit inspection")
                            request = read(completed_p[0].parent / "request.json")
                            if request["source_checkpoint"] != before["checkpoint"] or request["config"] != config:
                                raise ValueError("Completed P-step belongs to a different source/config")
                            p = read(completed_p[0])
                        else:
                            p = op.train(before["checkpoint"], h["path"], train_rows, config, attempt(rd / "p_train"))
                        if p.get("simulated") and not test_only:
                            raise ValueError("Simulated training cannot publish a production pair")
                        if p["harness_revision"] != h["revision"] or not p["reload_verified"]:
                            raise ValueError("P-step returned an unverified checkpoint/harness binding")
                    else:
                        p = {
                            "checkpoint": before["checkpoint"],
                            "harness_revision": h["revision"],
                            "skipped": True,
                            "statistics": None,
                        }
                    write(rd / "p_result.json", p)
                p = read(rd / "p_result.json")
                if not test_only:
                    check_pair(
                        {
                            "checkpoint": p["checkpoint"],
                            "harness": h,
                            "checkpoint_inspection": p.get("checkpoint_inspection", before["checkpoint_inspection"]),
                        }
                    )
                if not (rd / "new_eval.json").exists():
                    with op.serve(p["checkpoint"], config, attempt(rd / "serve_new_eval")) as client:
                        new_target = op.evaluate(
                            client, dev, p["checkpoint"], h["path"], config, attempt(rd / "new_new")
                        )
                        new_old = (
                            op.evaluate(
                                client, dev, p["checkpoint"], before["harness"]["path"], config, attempt(rd / "new_old")
                            )
                            if config["cross_eval"]
                            else None
                        )
                    write(rd / "new_eval.json", {"new_old": new_old, "new_new": new_target})
                cells = {**read(rd / "old_eval.json"), **read(rd / "new_eval.json")}
                write(rd / "eval_pairs.json", cells)
                metrics = {
                    "round": t,
                    "test_only": test_only,
                    "scores": {k: execution.mean_score(v) if v else None for k, v in cells.items()},
                    "candidate_count": len(h_result["history"]),
                    "accepted_candidates": sum(r["accepted"] for r in h_result["history"]),
                    "training": p["statistics"],
                    "inference": "descriptive paired observations; no internalization claim",
                }
                write(rd / "metrics.json", metrics)
                write(rd / "policy_after.json", p)
                completed = {
                    "round": t,
                    "pair": {
                        "checkpoint": p["checkpoint"],
                        "harness": h,
                        "checkpoint_inspection": p.get("checkpoint_inspection", before.get("checkpoint_inspection")),
                    },
                    "test_only": test_only,
                }
                write(rd / "completed.json", completed)
            except Exception as exc:
                write(rd / "failure.json", {"error_type": type(exc).__name__, "accepted_pair_unchanged": before})
                raise
        if not test_only:
            check_pair(completed["pair"])
        state.update(next_round=t + 1, pair=completed["pair"])
        write(output / "state.json", state)
    return state


def cross_eval(config, output, old_policy, new_policy, old_harness, new_harness):
    rows = configured_tasks(config, "dev_eval", config["dev_task_sample_size"])
    output = Path(output)
    if (output / "eval_pairs.json").exists():
        raise ValueError("Cross-eval output already exists")
    cells = {}
    for plabel, policy in [("old", old_policy), ("new", new_policy)]:
        with execution.serve(policy, config, output / f"serve_{plabel}") as client:
            for hlabel, harness in [("old", old_harness), ("new", new_harness)]:
                key = f"{plabel}_{hlabel}"
                cells[key] = execution.evaluate(client, rows, policy, harness, config, output / key)
    write(output / "eval_pairs.json", cells)
    return cells
