"""Tencent lifecycle around the official OSWorld environment and AGS proxies.

The explicit source patch only injects a Provider/VMManager pair. Agent actions,
task setup, reset validation and grading remain in the pinned OSWorld source.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from baseline.records import sha, write

from .official import path_value

COOKBOOK_COMMIT = "b83d476e5e31006570cd8fab9a55ee342ca41491"
PATCH = Path(__file__).with_name("osworld_provider.patch")
STREAM_PATCH = Path(__file__).with_name("osworld_ags_stream.patch")
RECORDING_PATCH = Path(__file__).with_name("osworld_recording_guest.patch")
SCREENSHOT_PATCH = Path(__file__).with_name("osworld_screenshot.patch")


def source(options):
    root = path_value(options["tencent"]["cookbook_repo"])
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if head != COOKBOOK_COMMIT or dirty:
        raise ValueError("Tencent cookbook requires clean tracked source at " + COOKBOOK_COMMIT)
    return {
        "repo": str(root),
        "commit": head,
        "provider_patch_sha256": sha(PATCH),
        "http_stream_patch_sha256": sha(STREAM_PATCH),
        "recording_guest_patch_sha256": sha(RECORDING_PATCH),
        "screenshot_patch_sha256": sha(SCREENSHOT_PATCH),
    }


def prepare_provider(expected, out):
    import desktop_env.providers

    provider_dir = Path(expected["repo"]) / "examples/osworld-ags/overlay/OSWorld/desktop_env/providers"
    target = Path(out) / "providers/ags"
    shutil.copytree(provider_dir / "ags", target, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", str(target / "provider.py"), str(STREAM_PATCH)],
        check=True,
        capture_output=True,
    )
    desktop_env.providers.__path__.append(str(target.parent))


def prepare_environment(request):
    """Import a per-episode source copy with explicit patches; upstream stays clean."""
    expected = request["options"]["tencent_source"]
    if source(request["options"]) != expected:
        raise ValueError("Tencent source/patch changed after planning")
    timeout = request["options"]["tencent"]["screenshot_timeout"]
    if not 10 <= timeout <= 120:
        raise ValueError("Set an explicit screenshot HTTP timeout between 10 and 120 seconds")
    os.environ["OSWORLD_SCREENSHOT_TIMEOUT"] = str(timeout)
    out = Path(request["output"]) / "native_source"
    out.mkdir()
    if "desktop_env" in sys.modules:
        raise RuntimeError("Tencent evaluation requires a fresh official worker process")
    package = out / "desktop_env"
    shutil.copytree(
        Path(request["source"]["repo"]) / "desktop_env", package, ignore=shutil.ignore_patterns("__pycache__")
    )
    for target, patch in (
        (package / "desktop_env.py", PATCH),
        (package / "recording_update/guest.py", RECORDING_PATCH),
        (package / "controllers/python.py", SCREENSHOT_PATCH),
    ):
        subprocess.run(["patch", "--batch", "--fuzz=0", str(target), str(patch)], check=True, capture_output=True)
    sys.path.insert(0, str(out))
    prepare_provider(expected, package)
    write(out / "source.json", {**expected, "patched_environment_sha256": sha(package / "desktop_env.py")})
    from desktop_env.desktop_env import DesktopEnv

    return DesktopEnv


def provider_pair(options, output):
    # Imported after prepare_environment extends the native provider search path.
    from desktop_env.providers.ags.manager import AGSVMManager
    from desktop_env.providers.ags.provider import AGSProvider
    from e2b import Sandbox

    cfg = options["tencent"]
    path = Path(output) / "sandbox_lifecycle.jsonl"

    def event(operation, **data):
        with path.open("a") as f:
            f.write(json.dumps({"time": time.time(), "operation": operation, **data}) + "\n")

    class TencentProvider(AGSProvider):
        def start_emulator(self, path_to_vm, headless, os_type=None):
            if self.sandbox is not None:
                raise RuntimeError("Refusing to reuse an unclosed OSWorld sandbox")
            self.screenshot_timeout = cfg["screenshot_timeout"]
            self.sandbox = Sandbox.create(
                template=cfg["template"],
                domain=cfg["domain"],
                api_key=os.environ[cfg["key_env"]],
                timeout=cfg["sandbox_timeout"],
                request_timeout=cfg["request_timeout"],
                secure=True,
                http_version="1.1",
                retries=0,
                validate_api_key=False,
            )
            self.sandbox_id = self.sandbox.sandbox_id
            event("created", sandbox_id=self.sandbox_id, template=cfg["template"])
            try:
                info = self.sandbox.get_info()
                if info.template_id != cfg["tool_id"]:
                    raise ValueError("Tencent tool ID differs from the recorded configuration")
                event(
                    "identified",
                    sandbox_id=self.sandbox_id,
                    tool_id=info.template_id,
                    cpu=info.cpu_count,
                    memory_mb=info.memory_mb,
                )
                self.envd_access_token = self.sandbox._envd_access_token
                if not self.envd_access_token:
                    raise ValueError("Tencent sandbox did not return an access token")
                self._start_local_proxies()
                self._wait_for_vm_ready(timeout=cfg["ready_timeout"])
                self._ensure_chromium_cdp_bridge()
            except BaseException:
                self.stop_emulator(path_to_vm)
                raise

        def revert_to_snapshot(self, path_to_vm, snapshot_name):
            # Native DesktopEnv calls start_emulator immediately after this.
            # The cookbook's original no-op reset is deliberately not inherited.
            self.stop_emulator(path_to_vm)
            event("reset", snapshot=snapshot_name)

        def stop_emulator(self, path_to_vm, *args, **kwargs):
            self._stop_local_proxies()
            if self.sandbox is not None:
                try:
                    self.sandbox.kill()
                except Exception:
                    event("cleanup", sandbox_id=self.sandbox_id, success=False)
                    raise
                event("cleanup", sandbox_id=self.sandbox_id, success=True)
                self.sandbox = None
                self.sandbox_id = self.envd_access_token = None

    return AGSVMManager(), TencentProvider()
