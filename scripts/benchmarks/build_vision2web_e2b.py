#!/usr/bin/env python
"""Build the pinned official Dockerfile in one disposable Tencent CPU sandbox."""

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baseline.benchmarks.vision2web import Vision2Web
from baseline.benchmarks.vision2web_e2b import connect, sdk_options
from baseline.credentials import load_credentials
from baseline.records import ROOT, sha, write


def main():
    from e2b import Sandbox

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="New tagged destination in the company TCR registry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry-auth-file", type=Path, required=True)
    parser.add_argument("--builder-image", required=True, help="TCR mirror of Kaniko debug, pinned by digest")
    parser.add_argument("--repo", default=str(ROOT.parent / "references/Vision2Web"))
    args = parser.parse_args()
    if not args.image.startswith("modelbest.tencentcloudcr.com/") or ":" not in args.image:
        parser.error("Expected an explicit tagged TCR destination")
    if "@sha256:" not in args.builder_image:
        parser.error("Pin the builder environment image by digest")
    if args.registry_auth_file.stat().st_mode & 0o077:
        parser.error("Registry credentials must be owner-only")
    source = Vision2Web().source({"official": {"repo": args.repo}})
    dockerfile = Path(source["repo"]) / "docker/Dockerfile"
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    load_credentials(["E2B_API_KEY"])
    cfg = dict(
        key_env="E2B_API_KEY",
        domain="ap-beijing.tencentags.com",
        template="code-agent-cfs2-new",
        connection="direct",
        request_timeout=60,
        sandbox_timeout=7200,
    )
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(name, None)
    report = dict(
        source=source,
        dockerfile_sha256=sha(dockerfile),
        image=args.image,
        builder_image=args.builder_image,
        cpu=4,
        memory="16Gi",
        gpu=0,
        status="RUNNING",
        created=False,
        cleanup=False,
        simulated=False,
    )
    write(out / "build.json", report)
    started = time.monotonic()
    sbx = None
    try:
        custom = dict(
            image=args.builder_image,
            imageRegistryType="enterprise",
            command=["/busybox/sh", "-c"],
            # SDK commands need /bin/bash and a root user even in this scratch image.
            # Ubuntu extraction replaces these bootstrap files during the build.
            args=[
                "/busybox/mkdir -p /bin /tmp /root /etc; "
                "printf '#!/busybox/sh\\nexec /busybox/sh \"$@\"\\n' > /bin/bash; "
                "/busybox/chmod 755 /bin/bash; "
                "printf 'root:x:0:0:root:/root:/bin/bash\\n' > /etc/passwd; "
                "printf 'root:x:0:\\n' > /etc/group; exec /mnt/envd -port 49983"
            ],
            resources={"cpu": "4", "memory": "16Gi"},
        )
        sbx = Sandbox.create(
            template=cfg["template"],
            timeout=cfg["sandbox_timeout"],
            secure=True,
            allow_internet_access=True,
            metadata={"x-custom-config": json.dumps(custom), "purpose": "vision2web-image-build"},
            **sdk_options(cfg),
        )
        report.update(created=True, sandbox_id=sbx.sandbox_id)
        write(out / "build.json", report)
        sbx = connect(report, cfg)
        sbx.commands.run(
            "/busybox/mkdir -p /kaniko/context /kaniko/.docker; /busybox/chmod 700 /kaniko/.docker",
            user="root",
            timeout=20,
        )
        sbx.files.write("/kaniko/context/Dockerfile", dockerfile.read_bytes(), user="root")
        sbx.files.write("/kaniko/.docker/config.json", args.registry_auth_file.read_bytes(), user="root")
        sbx.commands.run("/busybox/chmod 600 /kaniko/.docker/config.json", user="root", timeout=20)
        image = shlex.quote(args.image)
        # Kaniko always excludes /kaniko from snapshots. Keep credentials there,
        # outside the context; also exclude the injected Tencent control plane.
        script = f"""#!/busybox/sh
set -eu
export DOCKER_CONFIG=/kaniko/.docker
export SSL_CERT_DIR=/kaniko/ssl/certs
export SSL_CERT_FILE=/kaniko/ssl/certs/ca-certificates.crt
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/busybox
/kaniko/executor --context=dir:///kaniko/context --dockerfile=/kaniko/context/Dockerfile \\
  --destination={image} --digest-file=/kaniko/result-digest \\
  --label=org.opencontainers.image.revision={Vision2Web.commit} \\
  --ignore-path=/mnt --ignore-path=/kaniko --ignore-path=/busybox \\
  --snapshot-mode=full --cache=false
"""
        (out / "build.sh").write_text(script)
        sbx.files.write("/kaniko/build.sh", script, user="root")
        print("Builder sandbox created; progress is in " + str(out / "build.log"), flush=True)
        with (out / "build.log").open("w") as log:

            def stream(chunk):
                log.write(chunk)
                log.flush()

            sbx.commands.run(
                "/busybox/sh /kaniko/build.sh",
                user="root",
                timeout=5400,
                on_stdout=stream,
                on_stderr=stream,
            )
        report["image_digest"] = sbx.files.read("/kaniko/result-digest", user="root").strip()
        report["pinned_image"] = args.image + "@" + report["image_digest"]
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error_type=type(exc).__name__)
        # Commands contain paths only; never persist SDK request/authorization details.
        from e2b.sandbox.commands.command_handle import CommandExitException

        if isinstance(exc, CommandExitException):
            report.update(exit_code=exc.exit_code, command_stderr=exc.stderr[-4000:])
    finally:
        if sbx:
            try:
                sbx.kill()
                report["cleanup"] = True
            except Exception as exc:
                report["cleanup_error_type"] = type(exc).__name__
                report["status"] = "FAIL"
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write(out / "build.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
