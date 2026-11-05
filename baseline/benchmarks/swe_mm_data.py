"""Pinned SWE-bench Multimodal v2 data, with separate public and grader files."""

import argparse
import json
import os
from pathlib import Path

from baseline.records import append, read, sha, write

DATASET = "SWE-bench/SWE-bench_Multimodal"
REVISION = "4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5"
COUNTS = {"dev": 100, "test": 480}


def assets(row):
    value = row.get("image_assets") or {}
    return json.loads(value) if isinstance(value, str) else value


def public_row(row):
    # hints, reference patch, test patch/assets and test names are host-only.
    return {
        "task_id": row["instance_id"],
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "sandbox_image": row["image"],
        "problem_statement": row["problem_statement"],
        "issue_images": assets(row).get("problem_statement") or [],
    }


def prepare(root, split_rows, provenance):
    """Keep official membership/order; no random re-split of final test data."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    seen, splits = set(), {}
    for split in ("dev", "test"):
        rows = split_rows[split]
        ids = [r["instance_id"] for r in rows]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Duplicate task IDs within/across official splits")
        seen.update(ids)
        for row in rows:
            for field in ("eval_script", "log_parser", "eval_type", "FAIL_TO_PASS", "PASS_TO_PASS"):
                if field not in row:
                    raise ValueError(f"Missing official grading field: {field}")
            append(root / split / "public.jsonl", public_row(row))
            append(root / split / "grader.jsonl", row)
        splits[split] = {
            "count": len(rows),
            "task_ids": ids,
            "public_sha256": sha(root / split / "public.jsonl"),
            "grader_sha256": sha(root / split / "grader.jsonl"),
            "repositories": sorted({r["repo"] for r in rows}),
        }
    write(root / "manifest.json", {**provenance, "splits": splits, "task_id_overlap": []})
    return read(root / "manifest.json")


def load_split(root, split):
    if split not in COUNTS:
        raise ValueError("SWE-MM split must be dev or test")
    root = Path(root)
    manifest = read(root / "manifest.json")
    if manifest["dataset"] != DATASET or manifest["revision"] != REVISION:
        raise ValueError("Unexpected SWE-MM dataset revision")
    for name in ("public", "grader"):
        if sha(root / split / f"{name}.jsonl") != manifest["splits"][split][f"{name}_sha256"]:
            raise ValueError("SWE-MM dataset changed since preparation")
    rows = [json.loads(line) for line in (root / split / "public.jsonl").read_text().splitlines()]
    if [r["task_id"] for r in rows] != manifest["splits"][split]["task_ids"]:
        raise ValueError("SWE-MM split task membership changed")
    return rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New ignored local data directory")
    parser.add_argument("--parquet-dir", help="Use already downloaded official Parquet files; no network")
    args = parser.parse_args()
    import pyarrow.parquet as pq
    import requests

    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(root)
    # Download only the two small data tables (~11 MB), never images or models.
    raw = Path(args.parquet_dir) if args.parquet_dir else root.parent / (root.name + "-parquet")
    raw.mkdir(parents=True, exist_ok=True)
    rows, sources = {}, {}
    for split, count in COUNTS.items():
        path = raw / f"{split}-00000-of-00001.parquet"
        if not path.exists():
            if args.parquet_dir:
                raise FileNotFoundError(path)
            response = requests.get(
                f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{path.name}", timeout=120
            )
            response.raise_for_status()
            path.write_bytes(response.content)
            os.chmod(path, 0o600)
        rows[split] = pq.read_table(path).to_pylist()
        if len(rows[split]) != count:
            raise ValueError(f"Expected {count} {split} rows at the pinned revision")
        sources[split] = {"path": str(path), "sha256": sha(path)}
    manifest = prepare(root, rows, {"dataset": DATASET, "revision": REVISION, "version": "v2", "sources": sources})
    print(
        json.dumps(
            {"manifest": str(root / "manifest.json"), "counts": {s: v["count"] for s, v in manifest["splits"].items()}}
        )
    )


if __name__ == "__main__":
    main()
