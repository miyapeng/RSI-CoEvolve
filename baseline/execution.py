"""Benchmark episode dispatch and paired evaluation; SWE serving aliases retained."""

from pathlib import Path

from .benchmarks import get_benchmark
from .records import append, digest, write
from .serving import serve as serve
from .serving import serve_command as serve_command


def rollout(client, row, checkpoint, harness, config, output, seed):
    benchmark = get_benchmark(config)
    result = benchmark.run_episode(client, row, checkpoint, harness, config, output, seed)
    result["benchmark"] = benchmark.name
    write(Path(output) / "result.json", result)
    return result


def evaluate(client, rows, checkpoint, harness, config, output):
    output = Path(output)
    benchmark = get_benchmark(config)
    results = []
    for repeat in range(config["eval_repetitions"]):
        for row in rows:
            iid = benchmark.task_info(row).task_id
            seed = int(digest([config["seed"], iid, repeat])[:8], 16) % 2147483647
            r = rollout(client, row, checkpoint, harness, config, output / f"r{repeat}" / iid, seed)
            r["repetition"] = repeat
            append(output / "results.jsonl", r)
            results.append(r)
    return results


def mean_score(rows):
    if not rows or any(r["score"] is None for r in rows):
        raise ValueError("Cannot score empty or infrastructure-failed evaluations")
    return sum(r["score"] for r in rows) / len(rows)
