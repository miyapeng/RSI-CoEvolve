"""Explicit one-sandbox connectivity check. No model API, benchmark or training."""

import argparse
import getpass
import hashlib
import json
import os
import time
from pathlib import Path

from .config import load_config
from .live_smoke import fixture_sandbox
from .records import ROOT, read, write
from .tencent import valid_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/baseline_9b.yaml"))
    parser.add_argument("--image-id", default="open-feature_python-sdk_pr329")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = {**config["runtime"]["sandbox"], "connection": "direct"}
    mappings = settings["images"]
    matched = [value for name, value in mappings.items() if name.endswith(":" + args.image_id)]
    if len(matched) != 1 or not valid_image(matched[0]):
        parser.error("Choose one existing pinned image in the local image manifest")
    image = matched[0]["image"] + "@" + matched[0]["digest"]
    if not args.execute:
        print(json.dumps({"execute": False, "requested_image": image, "cpu": 1, "memory": "1Gi", "ttl": 600}))
        return 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("E2B_DOMAIN", "ap-beijing.tencentags.com")
    os.environ.setdefault("E2B_TEMPLATE", "code-agent-cfs2-new")
    if not os.environ.get("E2B_API_KEY"):
        os.environ["E2B_API_KEY"] = getpass.getpass("E2B_API_KEY (hidden): ")
    # Direct API and envd routing, without changing the user's persistent proxy configuration.
    bypass = ",".join(filter(None, [os.environ.get("NO_PROXY", ""), ".tencentags.com", "modelbest.tencentcloudcr.com"]))
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = bypass
    report = {
        "kind": "real_sandbox_connectivity",
        "simulated": False,
        "requested_image": image,
        "domain": os.environ["E2B_DOMAIN"],
        "template": os.environ["E2B_TEMPLATE"],
        "model_calls": 0,
        "benchmark": "NOT_RUN",
        "training": "NOT_RUN",
        "checks": {},
        "status": "RUNNING",
    }
    started = time.monotonic()
    write(out / "report.json", report)
    try:
        print("Creating one temporary Tencent sandbox...", flush=True)
        with fixture_sandbox(settings, image, out) as (box, _):
            report["sandbox_id"] = box.sandbox.sandbox_id
            report["checks"]["created"] = True
            print("Sandbox created; checking commands and files...", flush=True)
            echo = box.checked("printf 'RSI_SANDBOX_OK\\n'", 20)
            if echo["output"].strip() != "RSI_SANDBOX_OK":
                raise RuntimeError("Unexpected command output")
            report["checks"]["command"] = True
            payload = bytes(range(256)) + b"\x00rsi-file-roundtrip\xff"
            remote = "/testbed/probe.bin"
            box.sandbox.files.write(remote, payload, user="root", request_timeout=30)
            downloaded = bytes(box.sandbox.files.read(remote, format="bytes", user="root", request_timeout=30))
            if downloaded != payload:
                raise RuntimeError("Binary file roundtrip mismatch")
            (out / "downloaded.bin").write_bytes(downloaded)
            report["checks"]["binary_upload_download"] = True
            report["file_sha256"] = hashlib.sha256(downloaded).hexdigest()
            versions = box.checked(
                "for cmd in python3 node npm google-chrome chromium playwright playwright-cli claude; "
                'do if command -v "$cmd" >/dev/null 2>&1; then printf \'%s:present\\n\' "$cmd"; '
                "else printf '%s:missing\\n' \"$cmd\"; fi; done",
                20,
            )
            report["available_tools"] = dict(line.split(":", 1) for line in versions["output"].strip().splitlines())
            write(out / "report.json", report)
        report["checks"]["cleanup"] = read(out / "lifecycle.json")["cleanup"]
        if not report["checks"]["cleanup"]:
            raise RuntimeError("Cleanup was not confirmed")
        report["status"] = "PASS"
    except Exception as exc:
        report.update(
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc).replace(os.environ["E2B_API_KEY"], "[REDACTED]")[:2000],
        )
    finally:
        if (out / "lifecycle.json").is_file():
            report["lifecycle"] = read(out / "lifecycle.json")
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write(out / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
