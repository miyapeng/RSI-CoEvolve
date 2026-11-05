"""Process-local transport for the pinned Vision2Web Docker command surface.

The official agent, prompts and verifier remain in their upstream checkout.
Only the worker's PATH is changed. Unsupported Docker operations fail closed.
"""

import contextlib
import io
import json
import os
import posixpath
import shlex
import sys
import tarfile
import time
import uuid
from pathlib import Path

from baseline.records import read, write
from baseline.tencent import SDK_VERSION


def validate(options):
    cfg = options["tencent"]
    allowed = {
        "key_env",
        "domain",
        "template",
        "connection",
        "proxy_env",
        "cpu",
        "memory",
        "sandbox_timeout",
        "command_timeout",
        "request_timeout",
    }
    if set(cfg) - allowed:
        raise ValueError("Unknown Tencent setting; use key_env for credentials")
    if cfg["connection"] not in {"direct", "proxy"}:
        raise ValueError("Tencent connection must be direct or proxy")
    image = os.path.expandvars(options["sandbox_image"])
    import re

    if not re.fullmatch(r"modelbest\.tencentcloudcr\.com/[\w./:-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Vision2Web Tencent image must be a TCR reference pinned with @sha256")
    options["sandbox_image"] = image
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", cfg["key_env"]):
        raise ValueError("key_env must name an environment variable")
    for field in ("cpu", "sandbox_timeout", "command_timeout", "request_timeout"):
        if cfg[field] <= 0:
            raise ValueError(f"Tencent {field} must be positive")
    return cfg


def sdk_options(cfg):
    options = dict(
        api_key=os.environ[cfg["key_env"]],
        domain=cfg["domain"],
        request_timeout=cfg["request_timeout"],
        http_version="1.1",
        retries=0,
        validate_api_key=False,
    )
    if cfg["connection"] == "proxy":
        options["proxy"] = os.environ[cfg["proxy_env"]]
    return options


def connect(state, cfg):
    from e2b import Sandbox

    options = sdk_options(cfg)
    if cfg["connection"] == "direct":
        options["sandbox_url"] = f"https://49983-{state['sandbox_id']}.{cfg['domain']}"
    return Sandbox.connect(state["sandbox_id"], timeout=cfg["sandbox_timeout"], **options)


def audit(root, operation, **fields):
    # Deliberately never record argv, environment values, prompts or SDK errors.
    with (root / "transport.jsonl").open("a") as f:
        f.write(json.dumps({"time": time.time(), "operation": operation, **fields}) + "\n")


@contextlib.contextmanager
def transport(request):
    """Install a private executable shim and clean up even on native failures."""
    options = request["options"]
    cfg = dict(validate(options))
    proxy = os.environ[cfg["proxy_env"]] if cfg["connection"] == "proxy" else None
    if proxy is not None:
        # Keep the host SDK proxy out of official SandboxManager._get_proxy_env().
        cfg["proxy_env"] = "VISION2WEB_E2B_PROXY"
    root = Path(request["output"]) / "transport"
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    write(root / "settings.json", {"config": cfg, "image": options["sandbox_image"], "allowed_root": request["output"]})
    launcher = root / "docker"
    # Use this exact Conda interpreter; no dependency on the caller's PATH/PYTHONPATH.
    source = Path(__file__).resolve().parents[2]
    launcher.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(source)!r})\n"
        "from baseline.benchmarks.vision2web_e2b import main\nraise SystemExit(main())\n"
    )
    launcher.chmod(0o700)
    saved = dict(os.environ)
    try:
        os.environ["PATH"] = str(root) + os.pathsep + os.environ.get("PATH", "")
        os.environ["VISION2WEB_E2B_STATE"] = str(root)
        # Host cluster proxies are not reachable from Tencent containers. The
        # SDK proxy, when requested, is supplied explicitly rather than inherited.
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)
        if proxy is not None:
            os.environ[cfg["proxy_env"]] = proxy
        yield
    finally:
        try:
            cleanup(root)
        finally:
            os.environ.clear()
            os.environ.update(saved)


def cleanup(root):
    settings = read(root / "settings.json")
    failures = []
    for path in root.glob("sandbox-*.json"):
        state = read(path)
        if state["stopped"]:
            continue
        try:
            connect(state, settings["config"]).kill()
            state["stopped"] = True
            write(path, state)
            audit(root, "cleanup", sandbox_id=state["sandbox_id"], success=True)
        except Exception as exc:
            failures.append(type(exc).__name__)
            audit(root, "cleanup", sandbox_id=state["sandbox_id"], success=False, error_type=type(exc).__name__)
    if failures:
        raise RuntimeError("Tencent cleanup not confirmed; see transport.jsonl and sandbox TTL")


def checked_archive(data):
    """Check before the official host-side tar command sees untrusted output."""
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = archive.getmembers()
        names = [posixpath.normpath(m.name) for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate paths in remote archive")
        links = {posixpath.normpath(m.name) for m in members if m.issym() or m.islnk()}
        for member in members:
            if member.name.startswith("/") or member.mode & 0o6000:
                raise ValueError("Absolute paths and privileged file modes are forbidden in archives")
            tarfile.data_filter(member, "/safe-extraction-root")
            # The official caller extracts with system tar, so validate link
            # traversal before streaming bytes into it (including chained links).
            paths = [posixpath.normpath(member.name)]
            if member.issym():
                paths.append(posixpath.normpath(posixpath.join(posixpath.dirname(member.name), member.linkname)))
            elif member.islnk():
                paths.append(posixpath.normpath(member.linkname))
            if any(p.startswith(link + "/") for p in paths for link in links):
                raise ValueError("Archive traverses a link")
            if len(paths) > 1 and paths[1] in links:
                raise ValueError("Chained archive links are unsupported")
    return data


def local_path(raw, settings):
    path = Path(raw).resolve()
    if not path.is_relative_to(Path(settings["allowed_root"]).resolve()):
        raise ValueError("Docker copy path escapes the episode output directory")
    return path


def run_command(box, cfg, argv, *, env=None, cwd="/workspace", user="root", background=False, stdin=None):
    from e2b.sandbox.commands.command_handle import CommandExitException

    command = shlex.join(argv)
    upload = None
    if stdin is not None:
        upload = f"/tmp/v2w-stdin-{uuid.uuid4().hex}"
        box.files.write(upload, stdin, user="root")
        command += " < " + shlex.quote(upload)
    try:
        if not background:
            return run_foreground(box, cfg, command, env or {}, cwd, user)
        return box.commands.run(
            command,
            envs=env or {},
            cwd=cwd,
            user=user,
            background=background,
            timeout=cfg["sandbox_timeout"] if background else cfg["command_timeout"],
            request_timeout=cfg["request_timeout"],
        )
    except CommandExitException as exc:
        return exc
    finally:
        if upload is not None:
            box.commands.run("rm -f " + shlex.quote(upload), user="root", timeout=10)


def run_foreground(box, cfg, command, env, cwd, user):
    """Keep long silent commands off the AGS streaming connection.

    Preserve stdout/stderr/exit status in private temporary files. No command
    retries: a failed poll fails the episode and its owner cleans up the sandbox.
    GNU timeout enforces the process deadline even if the client disconnects.
    """
    from e2b.sandbox.commands.command_handle import CommandResult

    root = "/tmp/v2w-command-" + uuid.uuid4().hex
    box.commands.run(f"mkdir -m 700 {root}", user=user, timeout=15)
    script = (
        "#!/bin/bash\n"
        f"timeout -k 5 {cfg['command_timeout']} bash -c {shlex.quote(command)} "
        f"> {root}/stdout 2> {root}/stderr\n"
        f"printf '%s' \"$?\" > {root}/exit.tmp\n"
        f"mv {root}/exit.tmp {root}/exit\n"
    )
    handle = None
    complete = False
    try:
        box.files.write(root + "/run.sh", script, user=user)
        handle = box.commands.run(
            "bash " + root + "/run.sh",
            background=True,
            envs=env,
            cwd=cwd,
            user=user,
            timeout=cfg["command_timeout"] + 10,
        )
        handle.disconnect()  # Closing this stream does not stop the remote process.
        deadline = time.monotonic() + cfg["command_timeout"] + 10
        while time.monotonic() < deadline:
            result = box.commands.run(
                f"if test -f {root}/exit; then cat {root}/exit; else printf RUNNING; fi",
                user=user,
                timeout=min(20, cfg["request_timeout"]),
            )
            if result.stdout != "RUNNING":
                complete = True
                return CommandResult(
                    stdout=box.files.read(root + "/stdout", user=user),
                    stderr=box.files.read(root + "/stderr", user=user),
                    exit_code=int(result.stdout),
                    error=None,
                )
            time.sleep(1)
        raise TimeoutError("Remote command exceeded its deadline; sandbox cleanup required")
    finally:
        if handle is not None and not complete:
            handle.kill()
        box.commands.run("rm -rf " + root, user=user, timeout=15)


def remote_archive(box, cfg, directory, name="."):
    remote = f"/tmp/v2w-transfer-{uuid.uuid4().hex}.tar"
    try:
        result = run_command(box, cfg, ["tar", "-C", directory, "-cf", remote, name], cwd="/")
        if result.exit_code:
            raise RuntimeError("Remote archive failed")
        return checked_archive(bytes(box.files.read(remote, format="bytes", user="root")))
    finally:
        box.commands.run("rm -f " + shlex.quote(remote), user="root", timeout=10)


def parse_flags(args, value_flags, switches):
    values, flags = {}, set()
    i = 0
    while i < len(args) and args[i].startswith("-"):
        flag = args[i]
        if flag in switches:
            flags.add(flag)
            i += 1
        elif flag in value_flags and i + 1 < len(args):
            values.setdefault(flag, []).append(args[i + 1])
            i += 2
        else:
            raise ValueError("Unsupported Docker option")
    return values, flags, args[i:]


def dispatch(root, args, stdin, stdout, stderr):
    from e2b import Sandbox

    settings = read(root / "settings.json")
    cfg = settings["config"]
    op, *args = args
    audit(root, op)
    if op == "images" and args == ["-q", settings["image"]]:
        # This means a configured remote reference, not a local Docker pull.
        stdout.write(settings["image"].split("@sha256:")[1].encode() + b"\n")
        return 0
    if op == "create":
        values, flags, tail = parse_flags(args, {"--name", "--workdir", "--memory", "--memory-swap", "-e"}, {"--rm"})
        if tail != [settings["image"], "sleep", "infinity"] or values.get("--workdir") != ["/workspace"]:
            raise ValueError("Unexpected official create command")
        env = dict(x.split("=", 1) for x in values.get("-e", []))
        custom = {
            "image": settings["image"],
            "imageRegistryType": "enterprise",
            "command": ["/bin/sh", "-c"],
            "args": ["/mnt/envd -port 49983"],
            "resources": {"cpu": str(cfg["cpu"]), "memory": cfg["memory"]},
        }
        box = Sandbox.create(
            template=cfg["template"],
            timeout=cfg["sandbox_timeout"],
            secure=True,
            allow_internet_access=True,
            metadata={"x-custom-config": json.dumps(custom)},
            **sdk_options(cfg),
        )
        state = {
            "sandbox_id": box.sandbox_id,
            "name": values["--name"][0],
            "stopped": False,
            "environment": env,
            "requested_memory": values.get("--memory"),
            "resources": custom["resources"],
        }
        try:
            write(root / f"sandbox-{box.sandbox_id}.json", state)
        except BaseException:
            box.kill()
            raise
        audit(root, "created", sandbox_id=box.sandbox_id, resources=custom["resources"])
        stdout.write(box.sandbox_id.encode() + b"\n")
        return 0
    if op == "ps":
        if (
            len(args) != 4
            or args[0] != "--filter"
            or not args[1].startswith("name=")
            or args[2:] != ["--format", "{{.ID}}"]
        ):
            raise ValueError("Unsupported Docker ps")
        for path in root.glob("sandbox-*.json"):
            state = read(path)
            if not state["stopped"] and args[1][5:] in state["name"]:
                stdout.write(state["sandbox_id"].encode() + b"\n")
        return 0
    values, flags = {}, set()
    if op == "exec":
        values, flags, args = parse_flags(args, {"-e", "-w", "--user"}, {"-i", "-d"})
    if op not in {"start", "stop", "exec", "cp"}:
        raise ValueError("Unsupported Docker operation")
    if op in {"start", "stop"} and len(args) != 1:
        raise ValueError("Expected exactly one sandbox ID")
    remote_spec = next((a for a in args if ":/" in a), "") if op == "cp" else args[0]
    iid = remote_spec.split(":", 1)[0]
    if not iid or not iid.isalnum():
        raise ValueError("Invalid sandbox ID")
    path = root / f"sandbox-{iid}.json"
    state = read(path)  # No access to sandboxes outside this run.
    if state["stopped"]:
        raise ValueError("Sandbox already stopped")
    box = connect(state, cfg)
    if op == "stop":
        box.kill()
        state["stopped"] = True
        write(path, state)
        audit(root, "stopped", sandbox_id=iid, success=True)
        return 0
    if op == "start":
        result = run_command(box, cfg, ["mkdir", "-p", "/workspace"], cwd="/")
        return result.exit_code
    if op == "cp":
        if len(args) != 2:
            raise ValueError("Unsupported Docker copy")
        source, destination = args
        if source.startswith(iid + ":"):
            remote = Path(source.split(":", 1)[1])
            target = local_path(destination, settings)
            target.mkdir(parents=True, exist_ok=True)
            data = remote_archive(box, cfg, str(remote.parent), remote.name)
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                archive.extractall(target, filter="data")
        else:
            local = local_path(source, settings)
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w") as archive:
                archive.add(local, arcname=".")
            checked_archive(data.getvalue())
            tmp = f"/tmp/v2w-upload-{uuid.uuid4().hex}.tar"
            box.files.write(tmp, data.getvalue(), user="root")
            dest = destination.split(":", 1)[1]
            try:
                result = run_command(
                    box,
                    cfg,
                    ["sh", "-c", f"mkdir -p {shlex.quote(dest)} && tar -C {shlex.quote(dest)} -xf {tmp}"],
                    cwd="/",
                )
                if result.exit_code:
                    raise RuntimeError("Remote upload extraction failed")
            finally:
                box.commands.run("rm -f " + tmp, user="root", timeout=10)
        return 0
    command = args[1:]
    if command[:2] == ["tar", "-C"]:
        if len(command) != 6 or command[3:] != ["-cf", "-", "."]:
            raise ValueError("Unsupported binary Docker exec")
        stdout.write(remote_archive(box, cfg, command[2]))
        return 0
    env = {**state["environment"], **dict(x.split("=", 1) for x in values.get("-e", []))}
    result = run_command(
        box,
        cfg,
        command,
        env=env,
        cwd=values.get("-w", ["/workspace"])[0],
        user=values.get("--user", ["root"])[0],
        background="-d" in flags,
        stdin=stdin.read() if "-i" in flags else None,
    )
    if "-d" in flags:
        return 0
    stdout.write(result.stdout.encode())
    stderr.write(result.stderr.encode())
    return result.exit_code


def main():
    import importlib.metadata

    if importlib.metadata.version("e2b") != SDK_VERSION:
        raise RuntimeError(f"Expected e2b=={SDK_VERSION}")
    root = Path(os.environ["VISION2WEB_E2B_STATE"])
    try:
        return dispatch(root, sys.argv[1:], sys.stdin.buffer, sys.stdout.buffer, sys.stderr.buffer)
    except Exception as exc:
        # Expose this actionable provider status without logging arbitrary SDK
        # messages, which may include authorization or signed request details.
        preheating = "ResourceUnavailable.ImagePreheating" in str(exc)
        audit(root, "error", error_type=type(exc).__name__, image_preheating=preheating)
        hint = "image is preheating; wait before a new attempt" if preheating else "see transport.jsonl"
        print(f"Vision2Web Tencent transport failed: {type(exc).__name__}; {hint}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
