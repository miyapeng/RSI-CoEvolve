"""Summarize observed records only; absent metrics remain empty/NOT_RUN."""

import csv
from pathlib import Path

from .records import read


def summarize(run):
    run = Path(run)
    out = run / "summaries"
    out.mkdir(parents=True, exist_ok=True)
    rounds, candidates, pairs = [], [], []
    for rd in sorted((run / "rounds").glob("round_*")):
        if (rd / "metrics.json").exists():
            m = read(rd / "metrics.json")
            rounds.append(
                {
                    "round": m["round"],
                    "test_only": m["test_only"],
                    **m["scores"],
                    "candidates": m["candidate_count"],
                    "accepted": m["accepted_candidates"],
                    "gradient_steps": (m["training"] or {}).get("observed_finite_nonzero_gradient_steps"),
                    "loss_tokens": (m["training"] or {}).get("effective_loss_tokens"),
                    "mixed_reward_group_fraction": (m["training"] or {}).get("mixed_reward_group_fraction"),
                }
            )
        if (rd / "h_result.json").exists():
            for c in read(rd / "h_result.json")["history"]:
                candidates.append(
                    {
                        "round": rd.name,
                        **{
                            k: c.get(k)
                            for k in (
                                "candidate",
                                "parent_revision",
                                "candidate_revision",
                                "accepted",
                                "rejection",
                                "mean_tokens",
                            )
                        },
                    }
                )
        if (rd / "eval_pairs.json").exists():
            for cell, rows in read(rd / "eval_pairs.json").items():
                for row in rows or []:
                    pairs.append(
                        {
                            "round": rd.name,
                            "cell": cell,
                            **{
                                k: row.get(k)
                                for k in (
                                    "task_id",
                                    "split",
                                    "policy_checkpoint",
                                    "harness_revision",
                                    "sampling_seed",
                                    "score",
                                    "tokens",
                                    "tool_calls",
                                    "latency_seconds",
                                    "error_type",
                                )
                            },
                        }
                    )
    for name, values in [("round_metrics", rounds), ("candidate_metrics", candidates), ("paired_evaluation", pairs)]:
        with (out / f"{name}.csv").open("w") as f:
            if values:
                writer = csv.DictWriter(f, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    status = "TEST FIXTURE ONLY" if any(r["test_only"] for r in rounds) else ("OBSERVED" if rounds else "NOT_RUN")
    text = f"# Baseline run\n\nStatus: {status}. Completed rounds: {len(rounds)}.\n\n"
    text += (
        "| Round | old policy/old H | old policy/new H | new policy/old H | new policy/new H |\n|---|---|---|---|---|\n"
    )
    for r in rounds:
        text += f"| {r['round']} | {r['old_old']} | {r['old_new']} | {r['new_old']} | {r['new_new']} |\n"
    text += "\nDev is adaptively reused. Small samples are statistically uncertain. These observations do not establish internalization.\n"
    (out / "report.md").write_text(text)
    return str(out / "report.md")
