"""Shared task preparation over Tencent E2B; separate agent and grader sandboxes."""

import shlex
from types import SimpleNamespace

from .data import public_task
from .tencent import TencentSandbox


class TaskSandbox:
    def __init__(self, task, config, *, grader=False, pre_commands=""):
        self.task, self.settings, self.grader = task, config, grader
        self.config = SimpleNamespace(cwd=task["cwd"])
        self.sandbox = None
        self._transport = TencentSandbox(task, config, grader=grader)
        self.pre_commands = pre_commands

    def __enter__(self):
        try:
            self._transport.create()
            self.sandbox = self._transport.sandbox
            self.execution_image = self._transport.image
            self.registry_digest = self._transport.registry_digest
            self.prepare()
            return self
        except BaseException:
            self.__exit__()
            raise

    def prepare(self):
        """Common preparation for a task image or an explicitly labelled coding probe."""
        cwd, base = shlex.quote(self.task["cwd"]), shlex.quote(self.task["base_commit"])
        self.checked(f"git -C {cwd} -c safe.directory={cwd} reset --hard {base}", self.settings["setup_timeout"])
        if self.pre_commands:
            self.checked(self.pre_commands, self.settings["setup_timeout"])
        if not self.grader:
            # Keep the seed's git-diff submission protocol, but remove access to source history.
            isolation = self.checked(
                f"""set -e
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
""",
                self.settings["setup_timeout"],
            )
            if isolation["output"].strip().splitlines()[-1] != "65534":
                raise RuntimeError("Task image did not enforce the unprivileged agent UID")

    def raw(self, command, timeout=120, cwd=None):
        return self._transport.raw(command, timeout, cwd)

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
        self._transport.upload(path, text)

    def __exit__(self, *_):
        self._transport.close()


class GraderEnvironment:
    """Explicit environment injection into unchanged benchmark grading operations."""

    def __init__(self, task, config, **upstream):
        if upstream["image"] != task["image"]:
            raise ValueError("Verifier requested an unexpected task image")
        self.cwd = upstream.get("cwd", task["cwd"])
        settings = {
            **config,
            "cpu": upstream.get("cpu", config["cpu"]),
            "memory": upstream.get("memory", config["memory"]),
        }
        self.task_sandbox = TaskSandbox(task, settings, grader=True).__enter__()

    def execute(self, action, cwd="", timeout=120):
        return self.task_sandbox.raw(action["command"], timeout, cwd=cwd or self.cwd)

    def cleanup(self):
        self.task_sandbox.__exit__()


def open_task(row, config, *, grader=False):
    pre = row["metadata"].get("pre_commands", "").replace("\\n", "\n").strip()
    return TaskSandbox(public_task(row), config, grader=grader, pre_commands=pre)
