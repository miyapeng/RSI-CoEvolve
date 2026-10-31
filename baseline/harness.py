"""Immutable code/config snapshots and a deliberately small host-side harness API."""

import ast
import builtins
import difflib
import json
import re
import shutil
from pathlib import Path

import yaml
from minisweagent.exceptions import Submitted
from minisweagent.run.benchmarks.utils.runner_common import ProgressTrackingAgent

from .records import digest

FILES = ("agent.py", "config.yaml", "MARKER.txt", "ACT_MARKER.txt", "purpose")


def revision(path):
    path = Path(path)
    return digest({name: (path / name).read_text() for name in FILES if (path / name).exists()})


def snapshot(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if revision(source) != revision(destination):
            raise ValueError("Refusing to replace an existing harness snapshot")
    else:
        destination.mkdir(parents=True)
        for name in FILES:
            if (source / name).exists():
                shutil.copyfile(source / name, destination / name)
    return {"path": str(destination.resolve()), "revision": revision(destination)}


def validate_code(text):
    tree = ast.parse(text)
    # A constrained Python hook surface, not unrestricted execution of API-generated host code.
    # Task-side shell commands still run only in Orchard, under a non-root uid.
    safe_names = {
        "super",
        "len",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "range",
        "enumerate",
        "zip",
        "min",
        "max",
        "sum",
        "any",
        "all",
        "sorted",
        "isinstance",
        "Submitted",
    }
    safe_methods = {
        "get",
        "items",
        "keys",
        "values",
        "append",
        "extend",
        "insert",
        "pop",
        "copy",
        "update",
        "strip",
        "lstrip",
        "rstrip",
        "split",
        "splitlines",
        "join",
        "startswith",
        "endswith",
        "replace",
        "lower",
        "upper",
        "search",
        "findall",
        "sub",
        "loads",
        "dumps",
    }
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            raise ValueError("Harness module may contain imports and AgentHarness only")
        if isinstance(node, ast.Import) and any(a.name not in ("re", "json") or a.asname for a in node.names):
            raise ValueError("Only re/json imports are allowed in candidate hooks")
        if isinstance(node, ast.ImportFrom):
            expected = {
                "minisweagent.run.benchmarks.swebench": "ProgressTrackingAgent",
                "minisweagent.exceptions": "Submitted",
            }
            if len(node.names) != 1 or node.module not in expected or node.names[0].name != expected[node.module]:
                raise ValueError("Unsupported harness import")
        if isinstance(node, ast.ClassDef):
            if node.name != "AgentHarness" or node.decorator_list or node.keywords or len(node.bases) != 1:
                raise ValueError("Expected one plain AgentHarness class")
            if not isinstance(node.bases[0], ast.Name) or node.bases[0].id != "ProgressTrackingAgent":
                raise ValueError("AgentHarness must inherit the real ProgressTrackingAgent")
            for fn in node.body:
                if not isinstance(fn, ast.FunctionDef) or fn.decorator_list or fn.returns:
                    raise ValueError("Harness class contains only undecorated methods")
                if fn.name not in ("step", "execute_actions") and not fn.name.startswith("_hook_"):
                    raise ValueError("Editable hooks: step/execute_actions and _hook_* helpers")
                if fn.args.defaults or fn.args.kw_defaults or any(a.annotation for a in fn.args.args):
                    raise ValueError("No evaluated defaults or annotations in candidate methods")
    for n in ast.walk(tree):
        if isinstance(n, (ast.Global, ast.Nonlocal, ast.With, ast.AsyncFunctionDef, ast.Lambda, ast.While)):
            raise ValueError("Unsupported host-side construct")
        if isinstance(n, ast.Name) and n.id.startswith("__"):
            raise ValueError("Host introspection is not allowed")
        if isinstance(n, ast.Attribute):
            if n.attr.startswith("__"):
                raise ValueError("Host introspection is not allowed")
            if isinstance(n.value, ast.Name) and n.value.id == "self":
                allowed = {"env", "messages", "repo_dir", "repo_diff", "add_messages", "mark", "n_calls", "deadline"}
                if n.attr not in allowed and not n.attr.startswith("_hook_"):
                    raise ValueError(f"Unsupported self attribute: {n.attr}")
                if isinstance(n.ctx, ast.Store) and not n.attr.startswith("_hook_"):
                    raise ValueError("Core agent state is immutable to hook assignment")
            elif n.attr.startswith("_"):
                raise ValueError("Private host attributes are not allowed")
            elif n.attr not in safe_methods | {"execute", "step", "execute_actions"}:
                raise ValueError("Hook attribute access is limited to the public data/tool API")
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id not in safe_names:
                raise ValueError(f"Unsupported host call: {n.func.id}")
            if isinstance(n.func, ast.Attribute):
                f = n.func
                if isinstance(f.value, ast.Name) and f.value.id == "self":
                    ok = f.attr in ("mark", "add_messages", "repo_diff") or f.attr.startswith("_hook_")
                elif ast.unparse(f.value) == "self.env":
                    ok = f.attr == "execute"
                elif ast.unparse(f.value) == "super()":
                    ok = f.attr in ("step", "execute_actions")
                else:
                    ok = f.attr in safe_methods
                if not ok:
                    raise ValueError(f"Unsupported host method: {ast.unparse(f)}")
    return tree


def check_diagnostic_leak(code, rows):
    """Reject direct memorization; this is not a proof against semantic contamination."""
    for row in rows:
        if row["metadata"]["instance_id"] in code or (row["patch"].strip() and row["patch"].strip() in code):
            raise ValueError("Candidate embeds training task IDs or a reference patch")
        paths = re.findall(r"^\+\+\+ b/(.+)$", row["patch"], flags=re.M)
        if any(path.strip() in code for path in paths):
            raise ValueError("Candidate hard-codes a source path from offline reference-patch diagnosis")


def load(path, expected=None):
    path = Path(path)
    if expected and revision(path) != expected:
        raise ValueError("Frozen harness changed")
    text = (path / "agent.py").read_text()
    seed = "from minisweagent.run.benchmarks.swebench import ProgressTrackingAgent as AgentHarness"
    if text.strip() == seed:
        cls = ProgressTrackingAgent
    else:
        tree = validate_code(text)
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        names = (
            "__build_class__",
            "super",
            "len",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "range",
            "enumerate",
            "zip",
            "min",
            "max",
            "sum",
            "any",
            "all",
            "sorted",
            "isinstance",
        )
        scope = {
            "__builtins__": {n: getattr(builtins, n) for n in names},
            "__name__": "baseline_candidate",
            "ProgressTrackingAgent": ProgressTrackingAgent,
            "Submitted": Submitted,
            "re": re,
            "json": json,
        }
        exec(compile(tree, str(path / "agent.py"), "exec"), scope)
        cls = scope["AgentHarness"]
    return cls, yaml.safe_load((path / "config.yaml").read_text())


def diff(parent, candidate):
    return "".join(
        difflib.unified_diff(
            (Path(parent) / "agent.py").read_text().splitlines(True),
            (Path(candidate) / "agent.py").read_text().splitlines(True),
            fromfile="parent/agent.py",
            tofile="candidate/agent.py",
        )
    )
