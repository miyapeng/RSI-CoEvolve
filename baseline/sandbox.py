"""Orchard transport; separate agent and grader sandboxes, no host data mounts."""

import os
import shlex
from types import SimpleNamespace

from .data import public_task


class TaskSandbox:
    def __init__(self, task, config, *, grader=False, pre_commands=""):
        self.task, self.settings, self.grader = task, config, grader
        self.config = SimpleNamespace(cwd=task["cwd"])
        self.client = self.sandbox = None
        self.pre_commands = pre_commands

    def __enter__(self):
        from orchard_env import SandboxClient

        c = self.settings
        self.client = SandboxClient(base_url=os.environ[c["endpoint_env"]], api_key=os.environ[c["key_env"]])
        try:
            self.sandbox = self.client.create_sandbox(
                image=self.task["image"],
                block_network=not self.grader,
                cpu=str(c["cpu"]),
                memory=c["memory"],
                timeout=c["setup_timeout"],
            )
            self.sandbox.start_heartbeat(interval=60)
            cwd, base = shlex.quote(self.task["cwd"]), shlex.quote(self.task["base_commit"])
            self.checked(f"git -C {cwd} -c safe.directory={cwd} reset --hard {base}")
            if self.pre_commands:
                self.checked(self.pre_commands, self.settings["setup_timeout"])
            if not self.grader:
                if self.sandbox.get_network_config()["mode"] != "restricted":
                    raise RuntimeError("Agent sandbox network restriction is not active")
                # Keep the seed's git-diff submission protocol, but remove access to source history.
                self.checked(f"""set -e
mkdir -p /root/baseline-private /tmp/baseline-home
chmod 700 /root /root/baseline-private
mv {cwd}/.git /root/baseline-private/git
cd {cwd}
git init -q
git add -A
git -c user.name=baseline -c user.email=baseline@localhost commit -qm base
chown -R 65534:65534 {cwd} /tmp/baseline-home
if [ -d /tests ]; then chmod 700 /tests; fi
setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs --bounding-set=-all id -u
""")
            return self
        except BaseException:
            self.__exit__()
            raise

    def raw(self, command, timeout=120):
        result = self.sandbox.exec(command=command, cwd=self.task["cwd"], timeout=timeout, login_shell=False)
        if result.error or result.exit_code is None:
            raise RuntimeError("Orchard command failed at the infrastructure layer")
        return {
            "output": (result.stdout or "") + (result.stderr or ""),
            "returncode": result.exit_code,
            "exception_info": "",
        }

    def checked(self, command, timeout=120):
        result = self.raw(command, timeout)
        if result["returncode"]:
            raise RuntimeError(f"Sandbox setup/grading command failed with code {result['returncode']}")
        return result

    def execute(self, action, cwd="", timeout=60):
        if cwd and cwd != self.task["cwd"]:
            raise ValueError("Use shell cd for task subdirectories")
        command = action["command"]
        path = "/opt/conda/envs/testbed/bin:/opt/miniconda3/envs/testbed/bin:/usr/local/bin:/usr/bin:/bin"
        prefix = [
            "setpriv",
            "--reuid=65534",
            "--regid=65534",
            "--clear-groups",
            "--no-new-privs",
            "--bounding-set=-all",
            "env",
            "-i",
            f"PATH={path}",
            "HOME=/tmp/baseline-home",
            "LANG=C.UTF-8",
            "PAGER=cat",
            "bash",
            "-c",
            command,
        ]
        return self.raw(shlex.join(prefix), timeout)

    def upload(self, path, text):
        self.sandbox.upload_content(text.encode(), path)

    def __exit__(self, *_):
        try:
            if self.sandbox is not None:
                self.sandbox.delete()
        finally:
            if self.client is not None:
                self.client.close()


def open_task(row, config, *, grader=False):
    pre = row["metadata"].get("pre_commands", "").replace("\\n", "\n").strip()
    return TaskSandbox(public_task(row), config, grader=grader, pre_commands=pre)
