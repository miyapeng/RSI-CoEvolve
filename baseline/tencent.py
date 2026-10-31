"""Tencent AGS/E2B transport, adapted from the earlier local prototype.

No model API, agent loop or scoring code lives here. Registry credentials stay
on the host; task images are explicit tag/digest mappings, never guessed.
"""

import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess

SDK_VERSION = "2.62.0"


def required_env(config):
    keys = [config[k] for k in ("key_env", "template_env", "domain_env")]
    if config["connection"] == "proxy":
        keys.append(config["proxy_env"])
    return keys


def validate_config(config):
    allowed = {
        "backend",
        "key_env",
        "template_env",
        "domain_env",
        "proxy_env",
        "connection",
        "cpu",
        "memory",
        "setup_timeout",
        "sandbox_timeout",
        "request_timeout",
        "image_manifest",
        "images",
    }
    if set(config) - allowed:
        raise ValueError("Unknown sandbox setting; credentials must use environment-variable names")
    if config["backend"] != "tencent_e2b" or config["connection"] not in ("proxy", "direct"):
        raise ValueError("This baseline uses tencent_e2b with proxy or direct connection")
    for key in required_env(config):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key):
            raise ValueError("Expected an environment-variable name, not a credential")


def valid_image(mapping):
    return isinstance(mapping, dict) and bool(
        re.fullmatch(r"[A-Za-z0-9.-]+(?::[0-9]+)?/[A-Za-z0-9_./-]+:[A-Za-z0-9_.-]+", mapping.get("image") or "")
        and not mapping["image"].endswith(":latest")
        and re.fullmatch(r"sha256:[0-9a-f]{64}", mapping.get("digest") or "")
    )


def preflight(config, images=()):
    validate_config(config)
    try:
        version = importlib.metadata.version("e2b")
    except importlib.metadata.PackageNotFoundError:
        version = None
    mappings = config.get("images", {})
    needed = set(images) if images else set(mappings)
    missing = sorted(needed - set(mappings))
    invalid = [name for name in sorted(needed & set(mappings)) if not valid_image(mappings[name])]
    blockers = [f"Tencent environment variable missing: {name}" for name in required_env(config) if not os.getenv(name)]
    if version != SDK_VERSION:
        blockers.append(f"Tencent transport requires e2b=={SDK_VERSION}")
    if not shutil.which("skopeo"):
        blockers.append("skopeo is required to verify mirrored image digests before creation")
    if not mappings or missing or invalid:
        blockers.append("Tencent image manifest is empty, incomplete or contains unpinned images")
    return {
        "backend": "tencent_e2b",
        "sdk": version,
        "blockers": blockers,
        "missing_images": missing,
        "invalid_images": invalid,
        "provider_network_isolation": "NOT_VERIFIED",
        "live_sandbox_test": "NOT_RUN",
    }


def check_image(mapping):
    result = subprocess.run(
        ["skopeo", "inspect", "--format", "{{.Digest}}", "docker://" + mapping["image"]],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(f"Registry inspection failed (exit {result.returncode}): {result.stderr.strip()[:2000]}")
    if result.stdout.strip() != mapping["digest"]:
        raise ValueError("Mirrored image tag has drifted from the recorded digest")


class TencentSandbox:
    def __init__(self, task, config, *, grader=False):
        self.task, self.config, self.grader = task, config, grader
        self.sandbox = None

    def create(self):
        from e2b import Sandbox

        check = preflight(self.config, [self.task["image"]])
        if check["blockers"]:
            raise ValueError("; ".join(check["blockers"]))
        cfg = self.config
        mapping = cfg["images"][self.task["image"]]
        check_image(mapping)
        self.image, self.registry_digest = mapping["image"], mapping["digest"]
        options = {
            "api_key": os.environ[cfg["key_env"]],
            "domain": os.environ[cfg["domain_env"]],
            "request_timeout": cfg["request_timeout"],
            "http_version": "1.1",
            "retries": 0,
            "validate_api_key": False,
        }
        if cfg["connection"] == "proxy":
            options["proxy"] = os.environ[cfg["proxy_env"]]
        custom = {
            "image": self.image,
            "imageRegistryType": "enterprise",
            "command": ["/bin/sh", "-c"],
            "args": ["/mnt/envd -port 49983"],
            "resources": {"cpu": str(cfg["cpu"]), "memory": cfg["memory"]},
        }
        self.sandbox = Sandbox.create(
            template=os.environ[cfg["template_env"]],
            timeout=cfg["sandbox_timeout"],
            secure=True,
            allow_internet_access=self.grader,
            metadata={"x-custom-config": json.dumps(custom), "purpose": self.task["split"]},
            **options,
        )
        try:
            if cfg["connection"] == "direct":
                # Class-level reconnect rebuilds envd routing. Keep the original
                # object until success so a failed reconnect can still be killed.
                self.sandbox = Sandbox.connect(
                    self.sandbox.sandbox_id,
                    timeout=cfg["sandbox_timeout"],
                    sandbox_url=f"https://49983-{self.sandbox.sandbox_id}.{options['domain']}",
                    **options,
                )
        except BaseException:
            self.close()
            raise

    def raw(self, command, timeout=120, cwd=None):
        from e2b.sandbox.commands.command_handle import CommandExitException

        command = "cd " + shlex.quote(cwd or self.task["cwd"]) + " || exit $?; " + command
        wrapped = shlex.join(["timeout", "-k", "5", str(timeout), "bash", "-c", command])
        chunks = []
        try:
            result = self.sandbox.commands.run(
                wrapped,
                user="root",
                timeout=timeout + 15,
                request_timeout=timeout + 30,
                on_stdout=chunks.append,
                on_stderr=chunks.append,
            )
        except CommandExitException as exc:
            result = exc  # A task exit code is distinct from SDK/network failure.
        if result.exit_code is None:
            raise RuntimeError("Tencent command returned no exit status")
        return {
            "output": "".join(chunks) or (result.stdout or "") + (result.stderr or ""),
            "returncode": result.exit_code,
            "exception_info": "",
        }

    def upload(self, path, text):
        self.sandbox.files.write(path, text, user="root")
        result = self.raw("chmod 600 " + shlex.quote(path))
        if result["returncode"]:
            raise RuntimeError("Cannot protect grader input permissions")

    def close(self):
        if self.sandbox is not None:
            sandbox, self.sandbox = self.sandbox, None
            sandbox.kill()
