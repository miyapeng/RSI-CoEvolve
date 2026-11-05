"""Inventory and mirror official SWE-MM task images, without starting sandboxes.

Requires skopeo and the user's REGISTRY_AUTH_FILE. Source manifests are pinned
before transfer; only verified destination digests enter the runtime image map.
"""

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from baseline.records import digest, read, write

from .swe_mm_data import COUNTS, REVISION, load_split


def records(dataset):
    return [row for split in COUNTS for row in load_split(dataset, split)[0]]


class RegistryRateLimit(RuntimeError):
    """Stop inventory requests when the upstream registry reports its quota."""


def inspect_raw(uri):
    result = subprocess.run(
        ["skopeo", "inspect", "--raw", "docker://" + uri], capture_output=True, timeout=120
    )
    if b"toomanyrequests" in result.stderr.lower() or b"pull rate limit" in result.stderr.lower():
        raise RegistryRateLimit("Docker Hub pull quota exhausted; supply registry authentication before resuming")
    result.check_returncode()
    return result.stdout


def inventory_one(source, root, offline=False):
    path = Path(root) / "images" / (digest(source) + ".json")
    if path.exists():
        cached = read(path)  # Never resolve a mutable source tag again on resume.
        if cached["source"] != source:
            raise ValueError("Cached manifest belongs to another image")
        return cached
    if offline:
        raise FileNotFoundError(path)
    raw = inspect_raw("docker.io/" + source)
    manifest = json.loads(raw)
    if "manifests" in manifest:
        amd64 = [
            m for m in manifest["manifests"]
            if m.get("platform", {}).get("architecture") == "amd64"
            and m["platform"].get("os") == "linux"
        ]
        if len(amd64) != 1:
            raise ValueError("Expected exactly one linux/amd64 manifest")
        raw = inspect_raw("docker.io/" + source.rsplit(":", 1)[0] + "@" + amd64[0]["digest"])
        if "sha256:" + hashlib.sha256(raw).hexdigest() != amd64[0]["digest"]:
            raise ValueError("Registry returned an inconsistent platform manifest")
        manifest = json.loads(raw)
    result = {
        "source": source,
        "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "platform": "linux/amd64",
        "layers": manifest["layers"],
        "config": manifest["config"],
    }
    write(path, result)
    return result


def inventory(dataset, root, workers=4, offline=False):
    sources = sorted({r["sandbox_image"] for r in records(dataset)})
    found, errors = [], {}
    # Read every existing pin first, including later tasks from a partial run.
    pending = []
    for source in sources:
        try:
            found.append(inventory_one(source, root, offline=True))
        except FileNotFoundError:
            pending.append(source)
    if offline:
        errors.update({s: "MissingPinnedManifest" for s in pending})
        pending = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Bounded batches let a rate limit stop the scan instead of queueing 580 requests.
        for start in range(0, len(pending), workers):
            futures = {executor.submit(inventory_one, s, root): s for s in pending[start:start + workers]}
            limited = False
            for future in as_completed(futures):
                source = futures[future]
                try:
                    found.append(future.result())
                except RegistryRateLimit:
                    errors[source] = "RegistryRateLimit"
                    limited = True
                except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                    errors[source] = type(exc).__name__
            print(json.dumps({"inventoried": len(found), "errors": len(errors), "total": len(sources)}), flush=True)
            if limited:
                errors.update({s: "NotInspectedAfterRateLimit" for s in pending[start + workers:]})
                break
    blobs = {b["digest"]: b["size"] for r in found for b in [r["config"], *r["layers"]]}
    report = {
        "dataset_revision": REVISION,
        "expected_images": len(sources),
        "inventoried": len(found),
        "errors": errors,
        "compressed_bytes_without_deduplication": sum(
            b["size"] for r in found for b in [r["config"], *r["layers"]]
        ),
        "unique_blob_bytes": sum(blobs.values()),
        "unique_blobs": len(blobs),
        "images": sorted(found, key=lambda r: r["source"]),
    }
    write(Path(root) / "image_inventory.json", report)
    return report


def registry_environment(registry):
    env = dict(os.environ)
    # Keep the cluster proxy for Docker Hub; the internal TCR route is direct.
    for key in ("NO_PROXY", "no_proxy"):
        env[key] = ",".join(filter(None, [env.get(key), registry]))
    return env


def mirror_one(row, entry, destination, root, existing):
    source = row["sandbox_image"]
    tag = existing.get(source, {}).get("image") or destination + ":v2-" + row["task_id"].lower()
    env = registry_environment(tag.split("/")[0])
    log = Path(root) / "transfers" / (digest(source) + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    # A prior successful mapping is still checked against the actual remote tag.
    if source in existing:
        actual = subprocess.check_output(
            ["skopeo", "inspect", "--format", "{{.Digest}}", "docker://" + tag], env=env, timeout=120, text=True
        ).strip()
        if actual == entry["digest"] == existing[source]["digest"]:
            return {"image": tag, "digest": actual}
        raise ValueError("Existing destination mapping changed; do not overwrite it")
    pinned = "docker.io/" + source.rsplit(":", 1)[0] + "@" + entry["digest"]
    with log.open("w") as stream:
        os.chmod(log, 0o600)
        result = subprocess.run(
            ["skopeo", "copy", "docker://" + pinned, "docker://" + tag],
            env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=7200,
        )
    if result.returncode:
        if "toomanyrequests" in log.read_text().lower() or "pull rate limit" in log.read_text().lower():
            raise RegistryRateLimit("Docker Hub pull quota exhausted")
        result.check_returncode()
    actual = subprocess.check_output(
        ["skopeo", "inspect", "--format", "{{.Digest}}", "docker://" + tag], env=env, timeout=120, text=True
    ).strip()
    if actual != entry["digest"]:
        raise ValueError("Source and destination image digests differ")
    return {"image": tag, "digest": actual}


def mirror(dataset, root, mapping_path, destination, workers=2, limit=None):
    root, mapping_path = Path(root), Path(mapping_path)
    inv = read(root / "image_inventory.json")
    if inv["dataset_revision"] != REVISION:
        raise ValueError("Image inventory dataset revision mismatch")
    entries = {r["source"]: r for r in inv["images"]}
    mapping = yaml.safe_load(mapping_path.read_text())["images"] if mapping_path.exists() else {}
    rows = records(dataset)
    # Previously configured tasks are verified first; then transfer in official order.
    rows = sorted(rows, key=lambda r: r["sandbox_image"] not in mapping)
    if limit is not None:
        rows = rows[:limit]
    errors, verified = {}, []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(rows), workers):
            futures, limited = {}, False
            for row in rows[start:start + workers]:
                source = row["sandbox_image"]
                if source not in entries:
                    errors[source] = "MissingPinnedManifest"
                    continue
                futures[executor.submit(mirror_one, row, entries[source], destination, root, dict(mapping))] = source
            for future in as_completed(futures):
                source = futures[future]
                try:
                    mapping[source] = future.result()
                    verified.append(source)
                    # JSON is also valid YAML; atomic replacement after each verified image.
                    write(mapping_path, {"images": mapping})
                except RegistryRateLimit:
                    errors[source], limited = "RegistryRateLimit", True
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    errors[source] = type(exc).__name__
                report = {"verified": verified, "errors": errors, "selected": len(rows), "configured": len(mapping)}
                write(root / "mirror_report.json", report)
                print(json.dumps({"verified": len(verified), "errors": len(errors), "selected": len(rows)}), flush=True)
            if limited:
                errors.update({r["sandbox_image"]: "NotMirroredAfterRateLimit" for r in rows[start + workers:]})
                break
    report = {"verified": verified, "errors": errors, "selected": len(rows), "configured": len(mapping)}
    write(root / "mirror_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "mirror"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-map", default="configs/swe_mm_images.local.yaml")
    parser.add_argument("--destination", default="modelbest.tencentcloudcr.com/code-interpreter/swe-mm")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offline", action="store_true", help="Summarize cached inventory only; no registry requests")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or (args.limit is not None and args.limit < 1):
        parser.error("Use 1..8 workers and a positive limit")
    if args.action == "inventory":
        report = inventory(args.dataset, args.output, args.workers, args.offline)
    else:
        if args.offline:
            parser.error("--offline is only valid for inventory")
        report = mirror(args.dataset, args.output, args.image_map, args.destination, args.workers, args.limit)
    print(json.dumps({k: v for k, v in report.items() if k not in ("images", "verified")}, indent=2))
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
