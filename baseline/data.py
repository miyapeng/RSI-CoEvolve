"""Local Self-HPO data validation and repository-grouped, source-stratified splits."""

import collections
import json
import random
from pathlib import Path

from .records import ROOT, digest, read, sha, write

ROLES = ("optimization_pool", "dev_eval", "audit_holdout")
DATASET = "self-hpo/Self-HPO-RL-Data"
REVISION = "321decd5b4bf3fd7fedfd3110d0b12071064af9f"


def source(row):
    m = row["metadata"]
    if m.get("dataset") == "AweAI-Team/Scale-SWE":
        return "Scale-SWE"
    if m.get("dataset") == "nebius/SWE-rebench-V2":
        return "SWE-rebench-V2"
    if not m.get("dataset") and m.get("install_config"):
        return "SWE-rebench (source field absent)"
    raise ValueError("Unknown training source; do not guess its grading semantics")


def repository(row):
    m = row["metadata"]
    repo = m["repo"]
    if source(row) == "Scale-SWE":
        repo = m["version"].removeprefix("https://github.com/").removesuffix(".git").rstrip("/")
    if "/" not in repo:
        raise ValueError(f"Cannot canonicalize repository: {m['instance_id']}")
    return repo.lower()


def load_raw(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = set()
    for r in rows:
        m = r["metadata"]
        for key in ("instance_id", "repo", "base_commit", "image_url"):
            if not isinstance(m.get(key), str) or not m[key]:
                raise ValueError(f"Missing task field {key}")
        if m["instance_id"] in ids:
            raise ValueError("Duplicate task ID")
        ids.add(m["instance_id"])
        if not r.get("problem_statement") or not isinstance(r.get("patch"), str):
            raise ValueError("Missing problem or reference patch")
        for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            if not isinstance(json.loads(m[k]) if isinstance(m[k], str) else m[k], list):
                raise ValueError(f"Invalid {k}")
        source(r)
        repository(r)
    return rows


def split(raw, output, seed=42):
    raw = Path(raw).absolute()
    rows = load_raw(raw)
    # Whole repositories stay together; identical patches across forks also stay together.
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, r in enumerate(rows):
        for key in [("repo", repository(r)), ("patch", digest(r["patch"]))]:
            if key[0] == "patch" and not r["patch"].strip():
                continue
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[find(i)].append(r)
    totals = collections.Counter(source(r) for r in rows)
    fractions = dict(zip(ROLES, (0.8, 0.1, 0.1), strict=True))
    counts = {role: collections.Counter() for role in ROLES}
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    entries = []
    for group in groups:
        sizes = collections.Counter(source(r) for r in group)

        # Greedy incremental squared-error objective against each source's target size.
        def penalty(role, sizes=sizes):
            return sum(
                (
                    (counts[role][s] + n - totals[s] * fractions[role]) ** 2
                    - (counts[role][s] - totals[s] * fractions[role]) ** 2
                )
                / totals[s]
                for s, n in sizes.items()
            )

        role = min(ROLES, key=penalty)
        counts[role].update(sizes)
        for r in group:
            m = r["metadata"]
            entries.append(
                {
                    "task_id": m["instance_id"],
                    "split": role,
                    "source": source(r),
                    "repository": repository(r),
                    "base_commit": m["base_commit"],
                    "image": m["image_url"],
                    "row_sha256": digest(r),
                    "patch_sha256": digest(r["patch"]),
                    "problem_sha256": digest(" ".join(r["problem_statement"].split())),
                }
            )
    overlaps = {}
    for key in ("task_id", "repository", "patch_sha256", "problem_sha256", "image"):
        by_key = collections.defaultdict(set)
        for e in entries:
            by_key[e[key]].add(e["split"])
        overlaps[key] = sum(len(v) > 1 for v in by_key.values())
    manifest = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "raw_path": str(raw.relative_to(ROOT)) if raw.is_relative_to(ROOT) else str(raw),
        "raw_sha256": sha(raw),
        "seed": seed,
        "requested_fractions": fractions,
        "strategy": "repository + exact patch connected components; source-stratified greedy assignment",
        "counts": {k: dict(v) for k, v in counts.items()},
        "cross_split_overlap": overlaps,
        "limitations": "Semantic near-duplicates not exhaustively detected; dev_eval is adaptively reused.",
        "tasks": sorted(entries, key=lambda e: e["task_id"]),
    }
    write(output, manifest)
    return manifest


def tasks(manifest_path, role, limit=None):
    if role not in ROLES:
        raise ValueError("Only training-source splits are supported here")
    manifest = read(manifest_path)
    raw_path = ROOT / manifest["raw_path"]
    if manifest["dataset"] != DATASET or sha(raw_path) != manifest["raw_sha256"]:
        raise ValueError("Training source changed or is not Self-HPO-RL-Data")
    entries = {e["task_id"]: e for e in manifest["tasks"] if e["split"] == role}
    result = []
    for row in load_raw(raw_path):
        entry = entries.get(row["metadata"]["instance_id"])
        if entry:
            if digest(row) != entry["row_sha256"]:
                raise ValueError("Task content differs from frozen manifest")
            result.append({**row, "split": role, "dataset_source": source(row)})
    result.sort(key=lambda r: digest([manifest["seed"], r["metadata"]["instance_id"]]))
    return result[:limit] if limit else result


def public_task(row):
    """The only task projection sent to execution. Gold/test metadata stay with the grader."""
    m = row["metadata"]
    return {
        "task_id": m["instance_id"],
        "problem_statement": row["problem_statement"],
        "image": m["image_url"],
        "cwd": m.get("workdir") or "/testbed",
        "base_commit": m["base_commit"],
        "split": row["split"],
        "dataset_source": row["dataset_source"],
    }
