# Adapted from Self-HPO slime/examples/mini_swe/swe_reward.py (MIT/Apache-2.0 notices upstream).
# Test-script construction is unchanged; importable without the RL/GPU stack.
import json
import shlex
from typing import Any, Optional
from swebench.harness.constants import LATEST, SWEbenchInstance, KEY_INSTANCE_ID, START_TEST_OUTPUT, END_TEST_OUTPUT
from .grading_report import GradingTestSpec
TestSpec = GradingTestSpec

def make_test_spec_scaleswe(
    instance: SWEbenchInstance,
    namespace: Optional[str] = None,
    base_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
    instance_image_tag: str = LATEST,
) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance.get("repo", "")
    version = instance.get("version")

    def _from_json_or_obj(key: str) -> Any:
        if key not in instance:
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS")
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS")

    test_patch = instance.get("f2p_patch") or instance.get("test_patch", "")

    repo_directory = instance.get("workdir", "/testbed")

    base_test_cmd = "pytest --no-header -rA --tb=line --color=no -p no:cacheprovider -W ignore::DeprecationWarning"
    all_test_ids = fail_to_pass + pass_to_pass
    if all_test_ids:
        test_id_str = " ".join(shlex.quote(tid) for tid in all_test_ids)
        test_command = f"{base_test_cmd} {test_id_str}"
    else:
        test_command = base_test_cmd

    install_config = {
        "env_vars": None,
        "env_yml_path": None,
        "install": "",
        "log_parser": "parse_log_pytest_scaleswe",
        "no_use_env": None,
        "packages": "",
        "pip_packages": [],
        "pre_install": None,
        "python": "3.11",
        "reqs_path": [],
        "test_cmd": test_command,
    }

    docker_specs = {}
    repo_script_list = []
    env_script_list = []
    HEREDOC_DELIMITER = "EOF_114329324912"

    eval_script_list = [
        f"cd {repo_directory}",
    ]
    if test_patch:
        eval_script_list.append(
            f"cat > /tmp/f2p_patch.diff <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
        eval_script_list.append(
            f"cd {repo_directory} && "
            f"(git apply --verbose /tmp/f2p_patch.diff || "
            f"git apply --verbose --reject /tmp/f2p_patch.diff || "
            f"patch --batch --forward --fuzz=5 -p1 -i /tmp/f2p_patch.diff || "
            f"true)"
        )
    eval_script_list.extend([
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
    ])

    arch = "x86_64"

    image_name = instance.get("image_url") or instance.get("image_name")

    return GradingTestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language='py',
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
        install_config=install_config,
        image_name=image_name,
    )


def make_test_spec_swerebench_v2(
    instance: SWEbenchInstance,
    namespace: Optional[str] = None,
    base_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
    instance_image_tag: str = LATEST,
) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"

    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance.get("repo", "")
    version = instance.get("version")

    def _from_json_or_obj(key: str) -> Any:
        if key not in instance:
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS")
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS")

    test_patch = instance.get("test_patch", "")
    repo_directory = instance.get("workdir", "/testbed")

    inst_install_config = instance.get("install_config", {})
    test_command = inst_install_config.get("test_cmd")
    log_parser = inst_install_config.get("log_parser", "parse_log_pytest")

    install_config = {
        "env_vars": None,
        "env_yml_path": None,
        "install": inst_install_config.get("install", ""),
        "log_parser": log_parser,
        "no_use_env": None,
        "packages": "",
        "pip_packages": [],
        "pre_install": None,
        "python": "3.10",
        "reqs_path": [],
        "test_cmd": test_command,
    }

    docker_specs = inst_install_config.get("docker_specs", {})
    repo_script_list = [install_config.get("install", "")]
    env_script_list = []
    HEREDOC_DELIMITER = "EOF_114329324912"

    _NON_TEST_EXTS = [".json", ".png", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".zip", ".csv", ".parquet", ".feather", ".svg", ".npy", ".yml", ".yaml"]
    _base_commit = instance.get("base_commit", "HEAD")
    _repo = instance.get("repo", "")
    def _changed_files(pp):
        return [l[6:] for l in (pp or "").split("\n") if l.startswith("+++ b/") and l[6:] not in ("", "/dev/null")]
    def _modified_files(pp):
        return [l[6:] for l in (pp or "").split("\n") if l.startswith("--- a/")]
    _directives = [d for d in _changed_files(test_patch) if not any(d.endswith(e) for e in _NON_TEST_EXTS)]
    if "pytest" in (test_command or ""):
        _directives = [d for d in _directives if d.endswith(".py")]
    if _repo == "django/django":
        _dt = []
        for d in _directives:
            d = d[:-3] if d.endswith(".py") else d
            d = d[6:] if d.startswith("tests/") else d
            _dt.append(d.replace("/", "."))
        _directives = _dt
    _test_full = " ".join([test_command, *_directives]).strip()
    _tfiles = _modified_files(test_patch)
    _reset = ("git checkout %s %s" % (_base_commit, " ".join(_tfiles))) if _tfiles else 'echo no-reset'
    _install = (install_config.get("install") or "").strip() or ": no-install"
    _pathprep = "export PATH=/opt/conda/envs/testbed/bin:/opt/miniconda3/envs/testbed/bin:$PATH"
    _act = "source /opt/miniconda3/bin/activate 2>/dev/null; source /opt/conda/bin/activate 2>/dev/null; conda activate testbed 2>/dev/null || true"
    _pybin = "PYBIN=$(ls /opt/conda/envs/testbed/bin/python /opt/miniconda3/envs/testbed/bin/python 2>/dev/null | head -1); [ -n \"$PYBIN\" ] || PYBIN=$(command -v python3 || command -v python)"
    _install_run = ("$PYBIN -m " + _install) if _install.startswith("pip ") else _install
    _test_run = ("$PYBIN -m " + _test_full) if _test_full.startswith("pytest") else _test_full
    eval_script_list = [
        "set -o pipefail",
        _pybin, _pathprep, _act,
        f"cd {repo_directory}",
        "export PYTHONPATH=$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}",
        f"git config --global --add safe.directory {repo_directory} 2>/dev/null || true",
        _install_run,
        _reset,
    ]
    if test_patch:
        eval_script_list.append(
            f"cat > /tmp/test_patch.diff <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
        eval_script_list.append(
            "git apply -v /tmp/test_patch.diff || git apply -v --reject /tmp/test_patch.diff || patch --batch --fuzz=5 -p1 -i /tmp/test_patch.diff || true"
        )
    eval_script_list += [
        f": '{START_TEST_OUTPUT}'",
        _test_run,
        f": '{END_TEST_OUTPUT}'",
        _reset,
    ]
    arch = "x86_64"

    image_name = instance.get("image_url") or instance.get("image_name")

    return GradingTestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language="py",
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
        install_config=install_config,
        image_name=image_name,
    )
