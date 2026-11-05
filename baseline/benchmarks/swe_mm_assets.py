"""Prepare public issue images and private verifier assets for both official splits.

This downloads static files only. Failed URLs remain explicit blockers; no
replacement images, data re-splits, model calls or sandbox creation are involved.
"""

import argparse
import io
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from baseline.records import digest, read, sha, write

from .swe_mm_data import COUNTS, REVISION, assets, load_split


def image_metadata(content):
    from PIL import Image

    with Image.open(io.BytesIO(content)) as image:
        if image.format not in ("PNG", "JPEG", "WEBP", "GIF"):
            raise ValueError("Unsupported issue image type")
        result = {"width": image.width, "height": image.height, "format": image.format.lower()}
        image.verify()
    return result


def asset_url(entry):
    return entry if isinstance(entry, str) else entry["url"]


def checked_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in relative:
        raise ValueError("Invalid asset path")
    target = Path(root) / path
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Asset path escapes cache")
    return target


def cached_file(root, record):
    if record["status"] != "ready":
        raise ValueError("Asset is not ready: " + record["url"])
    path = checked_path(root, record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError("Cached asset content changed: " + record["url"])
    return path


def fetch_one(url, root, visibility):
    import requests

    root = Path(root)
    record_path = root / visibility / "records" / (digest(url) + ".json")
    if record_path.exists():
        previous = read(record_path)
        if previous["status"] == "ready":
            cached_file(root, previous)
            return previous
    record = {"url": url, "status": "missing"}
    try:
        if not url.startswith("https://"):
            raise ValueError("Asset URL must use HTTPS")
        parsed = urlsplit(url)
        download_url = url
        if parsed.hostname == "github.com" and "/blob/" in parsed.path:
            # A GitHub file-view link names the same asset, but returns HTML.
            # Keep the original URL and record the exact raw-file resolution.
            download_url = "https://raw.githubusercontent.com" + parsed.path.replace("/blob/", "/", 1)
            record["url_resolution"] = "github_blob_to_raw"
        record["download_url"] = download_url
        content = bytearray()
        cap = (20 if visibility == "public" else 100) * 1024 * 1024
        with requests.get(download_url, timeout=(15, 60), stream=True) as response:
            response.raise_for_status()
            record["final_url"] = response.url
            record["content_type"] = response.headers.get("Content-Type")
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > cap:
                    raise ValueError("Asset exceeds configured byte limit")
        metadata = image_metadata(content) if visibility == "public" else {}
        relative = f"{visibility}/blobs/{digest(url)}"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".part")
        temp.write_bytes(content)
        os.chmod(temp, 0o600)
        temp.replace(path)
        record.update(status="ready", path=relative, sha256=sha(path), bytes=len(content), **metadata)
    except (requests.RequestException, ValueError, OSError) as exc:
        record["error_type"] = type(exc).__name__
        if isinstance(exc, requests.HTTPError):
            record["http_status"] = exc.response.status_code
        # No automatic alternate URL, omitted image, or placeholder success.
    write(record_path, record)
    return record


def prepare_assets(dataset, root, workers=8):
    dataset, root = Path(dataset), Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    rows, urls = {}, {"public": set(), "grader": set()}
    for split in COUNTS:
        load_split(dataset, split)  # Validate original prepared table hashes.
        rows[split] = [json.loads(line) for line in (dataset / split / "grader.jsonl").read_text().splitlines()]
        for row in rows[split]:
            declared = assets(row)
            urls["public"].update(asset_url(a) for a in declared.get("problem_statement") or [])
            for kind in ("patch", "test_patch"):
                urls["grader"].update(a["url"] for a in declared.get(kind) or [] if a.get("path"))
    manifests = {kind: {} for kind in urls}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_one, url, root, kind): kind for kind, values in urls.items() for url in sorted(values)
        }
        for n, future in enumerate(as_completed(futures), 1):
            record = future.result()
            manifests[futures[future]][record["url"]] = record
            if n % 50 == 0:
                print(json.dumps({"assets_processed": n, "total": len(futures)}), flush=True)
    for kind, values in manifests.items():
        write(root / kind / "manifest.json", {"dataset_revision": REVISION, "assets": dict(sorted(values.items()))})
    # Match the official task.repo.asset_path convention, host-only for grading.
    for split_rows in rows.values():
        for row in split_rows:
            for kind in ("patch", "test_patch"):
                for entry in assets(row).get(kind) or []:
                    if not entry.get("path"):
                        continue
                    record = manifests["grader"][entry["url"]]
                    if record["status"] != "ready":
                        continue
                    source = cached_file(root, record)
                    target = checked_path(root / "grader/task_repo", f"tasks/{row['instance_id']}/test_assets/{entry['path']}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if sha(target) != record["sha256"]:
                            raise ValueError("Conflicting private asset paths: " + str(target))
                    else:
                        os.link(source, target)
    report = {
        "dataset_revision": REVISION,
        "dataset_manifest_sha256": sha(dataset / "manifest.json"),
        "assets": {
            kind: {
                "expected": len(values),
                "ready": sum(r["status"] == "ready" for r in values.values()),
                "bytes": sum(r.get("bytes", 0) for r in values.values()),
                "failures": [r for r in values.values() if r["status"] != "ready"],
            }
            for kind, values in manifests.items()
        },
        "splits": {},
    }
    for split, split_rows in rows.items():
        blocked = {}
        for row in split_rows:
            declared = assets(row)
            expected = [("public", asset_url(a)) for a in declared.get("problem_statement") or []]
            expected += [
                ("grader", a["url"]) for k in ("patch", "test_patch") for a in declared.get(k) or [] if a.get("path")
            ]
            missing = [url for kind, url in expected if manifests[kind][url]["status"] != "ready"]
            if missing:
                blocked[row["instance_id"]] = missing
        report["splits"][split] = {"tasks": len(split_rows), "assets_ready": len(split_rows) - len(blocked), "blocked": blocked}
    write(root / "readiness.json", report)
    return report


def manifest(root, visibility):
    value = read(Path(root) / visibility / "manifest.json")
    if value["dataset_revision"] != REVISION:
        raise ValueError("Asset cache dataset revision mismatch")
    return value["assets"]


def public_files(root, public):
    records = manifest(root, "public")
    return [(cached_file(root, records[asset_url(a)]), records[asset_url(a)]) for a in public["issue_images"]]


def grader_repo(root, row):
    records = manifest(root, "grader")
    repo = Path(root) / "grader/task_repo"
    for kind in ("patch", "test_patch"):
        for entry in assets(row).get(kind) or []:
            if not entry.get("path"):
                continue
            record = records[entry["url"]]
            cached_file(root, record)
            path = checked_path(repo, f"tasks/{row['instance_id']}/test_assets/{entry['path']}")
            if sha(path) != record["sha256"]:
                raise ValueError("Private staged asset changed")
    return str(repo.resolve())


def copy_public_observations(root, public, output):
    from .official import file_reference

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    observations = [{"kind": "text", "text": public["problem_statement"]}]
    for index, (source, record) in enumerate(public_files(root, public)):
        target = output / f"issue_{index}.{record['format']}"
        shutil.copyfile(source, target)
        reference = file_reference(target)
        reference["metadata"].update({k: record[k] for k in ("url", "width", "height")})
        observations.append({"kind": "image", "artifact": reference})
    write(output / "observations.json", observations)
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("Use 1..16 download workers")
    report = prepare_assets(args.dataset, args.output, args.workers)
    print(json.dumps(report, indent=2))
    return 2 if any(a["failures"] for a in report["assets"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
