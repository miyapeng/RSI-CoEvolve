"""Offline setup contracts; synthetic assets are not benchmark measurements."""

import hashlib
import io
import json
from pathlib import Path

import pytest
import requests

from baseline.benchmarks import swe_mm_images
from baseline.benchmarks.swe_mm import download_observations
from baseline.benchmarks.swe_mm_assets import fetch_one, grader_repo, prepare_assets
from baseline.benchmarks.swe_mm_data import DATASET, REVISION, prepare, public_row
from baseline.records import read, write


def task(iid):
    return {
        "instance_id": iid, "repo": "fixture/repo", "base_commit": "a" * 40,
        "image": f"fixture/{iid}:latest", "problem_statement": "Public issue only",
        "image_assets": {
            "problem_statement": ["https://example.org/issue.png"],
            "test_patch": [{"path": "fixtures/private.png", "url": "https://example.org/private.png"}],
        },
        "eval_script": "PRIVATE", "log_parser": "fixture", "eval_type": "fixture",
        "FAIL_TO_PASS": ["PRIVATE"], "PASS_TO_PASS": [],
    }


def data(tmp_path):
    root = tmp_path / "data"
    prepare(root, {"dev": [task("x__x-1")], "test": [task("y__y-2")]}, {"dataset": DATASET, "revision": REVISION})
    return root


def test_complete_cache_is_reused_and_private_assets_stay_outside_observations(monkeypatch, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    content = io.BytesIO()
    Image.new("RGB", (2, 3), "red").save(content, format="PNG")
    calls = []

    class Response:
        headers = {"Content-Type": "image/png"}

        def __init__(self, url):
            self.url = url

        def __enter__(self):
            calls.append(self.url)
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b"PRIVATE_GRADER_BYTES" if self.url.endswith("private.png") else content.getvalue()

    monkeypatch.setattr(requests, "get", lambda url, **kw: Response(url))
    root, cache = data(tmp_path), tmp_path / "assets"
    report = prepare_assets(root, cache, workers=1)
    assert all(s["assets_ready"] == 1 for s in report["splits"].values())
    assert len(calls) == 2  # URL deduplication across tasks/splits, not task re-splitting.
    prepare_assets(root, cache, workers=1)
    assert len(calls) == 2
    github = "https://github.com/example/repo/blob/main/issue.png"
    resolved = fetch_one(github, cache, "public")
    assert resolved["url"] == github and resolved["url_resolution"] == "github_blob_to_raw"
    assert calls[-1] == "https://raw.githubusercontent.com/example/repo/main/issue.png"
    obs = download_observations(public_row(task("x__x-1")), tmp_path / "observations", cache)
    assert "PRIVATE" not in json.dumps(obs) and "private.png" not in json.dumps(obs)
    assert len(obs) == 2 and obs[1]["artifact"]["metadata"]["width"] == 2
    native = Path(grader_repo(cache, task("x__x-1"))) / "tasks/x__x-1/test_assets/fixtures/private.png"
    assert native.read_bytes() == b"PRIVATE_GRADER_BYTES"
    native.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        grader_repo(cache, task("x__x-1"))


def test_download_failure_is_not_a_successful_placeholder(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        response = requests.Response()
        response.status_code = 404
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(requests, "get", fail)
    record = fetch_one("https://example.org/missing.png", tmp_path, "public")
    assert record["status"] == "missing" and record["http_status"] == 404
    assert "sha256" not in record
    write(tmp_path / "public/manifest.json", {"dataset_revision": REVISION, "assets": {record["url"]: record}})
    with pytest.raises(ValueError, match="not ready"):
        download_observations({"problem_statement": "text", "issue_images": [record["url"]]}, tmp_path / "out", tmp_path)


def test_inventory_pins_amd64_and_counts_shared_layers_once(monkeypatch, tmp_path):
    root = data(tmp_path)
    raw = json.dumps({"config": {"digest": "config", "size": 10}, "layers": [{"digest": "shared", "size": 100}]}).encode()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    index = json.dumps({"manifests": [{"platform": {"os": "linux", "architecture": "amd64"}, "digest": digest}]}).encode()
    calls = []

    def inspect(uri):
        calls.append(uri)
        return raw if "@" in uri else index

    monkeypatch.setattr(swe_mm_images, "inspect_raw", inspect)
    report = swe_mm_images.inventory(root, tmp_path / "setup", workers=1)
    assert report["inventoried"] == 2 and report["unique_blob_bytes"] == 110
    assert report["compressed_bytes_without_deduplication"] == 220
    assert report["images"][0]["digest"] == digest
    swe_mm_images.inventory(root, tmp_path / "setup", workers=1)
    assert len(calls) == 4  # Resume uses pinned manifests, not mutable latest tags.


def test_failed_mirror_never_enters_runtime_mapping(monkeypatch, tmp_path):
    root = data(tmp_path)
    setup, mapping = tmp_path / "setup", tmp_path / "images.yaml"
    write(setup / "image_inventory.json", {
        "dataset_revision": REVISION,
        "images": [{"source": task(i)["image"], "digest": "sha256:" + "a" * 64} for i in ("x__x-1", "y__y-2")],
    })

    def mirror(row, entry, *args):
        if row["task_id"] == "y__y-2":
            raise ValueError("Digest mismatch")
        return {"image": "registry.example/swe-mm:dev", "digest": entry["digest"]}

    monkeypatch.setattr(swe_mm_images, "mirror_one", mirror)
    report = swe_mm_images.mirror(root, setup, mapping, "registry.example/swe-mm", workers=1)
    assert report["verified"] == [task("x__x-1")["image"]]
    assert task("y__y-2")["image"] not in read(mapping)["images"]
    assert report["errors"][task("y__y-2")["image"]] == "ValueError"


def test_registry_limit_stops_scanning_and_offline_report_does_not_retry(monkeypatch, tmp_path):
    root, setup, calls = data(tmp_path), tmp_path / "setup", []

    def limited(uri):
        calls.append(uri)
        raise swe_mm_images.RegistryRateLimit()

    monkeypatch.setattr(swe_mm_images, "inspect_raw", limited)
    report = swe_mm_images.inventory(root, setup, workers=1)
    assert len(calls) == 1
    assert sorted(report["errors"].values()) == ["NotInspectedAfterRateLimit", "RegistryRateLimit"]
    assert report["inventoried"] == 0
    offline = swe_mm_images.inventory(root, setup, offline=True)
    assert len(calls) == 1 and set(offline["errors"].values()) == {"MissingPinnedManifest"}
