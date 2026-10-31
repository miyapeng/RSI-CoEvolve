"""Self-HPO's source-specific P-step grading, without GPU imports or hidden retries."""

from pathlib import Path

from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT

from .data import source
from .grading_report import get_eval_report
from .grading_spec import make_test_spec_scaleswe, make_test_spec_swerebench_v2
from .records import append, write
from .sandbox import open_task


def specification(row):
    return (make_test_spec_scaleswe if source(row) == "Scale-SWE" else make_test_spec_swerebench_v2)(row["metadata"])


def grade(row, patch, config, output, *, sandbox_factory=open_task):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not patch.strip():
        result = {"score": 0.0, "resolved": False, "reason": "empty_patch", "executed_tests": False}
        write(output / "reward.json", result)
        return result
    spec = specification(row)
    with sandbox_factory(row, config["sandbox"], grader=True) as env:
        env.upload("/tmp/patch.diff", patch)
        applied = False
        # Preserve the Self-HPO P-step application ladder and source-specific test construction.
        for command in [
            "git apply --verbose /tmp/patch.diff",
            "git apply --verbose --reject /tmp/patch.diff",
            "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
        ]:
            apply = env.raw(command)
            append(output / "patch_application.jsonl", {"command": command, **apply})
            if apply["returncode"] == 0:
                applied = True
                break
        if not applied:
            result = {"score": 0.0, "resolved": False, "reason": "patch_apply_failed", "executed_tests": False}
        else:
            script = spec.eval_script.replace("locale-gen", "locale-gen en_US.UTF-8")
            env.upload("/root/eval.sh", script)
            prefix = "PYTHONPATH= " if "pylint" in spec.instance_id else ""
            test = env.raw(prefix + "/bin/bash /root/eval.sh 2>&1", config["grading_timeout"])
            log = output / "test_output.txt"
            log.write_text(test["output"])
            log.chmod(0o600)
            if test["returncode"] in (-1, 124, 137) or not all(
                marker in test["output"] for marker in (START_TEST_OUTPUT, END_TEST_OUTPUT)
            ):
                write(output / "reward.json", {"score": None, "error_type": "grading_incomplete"})
                raise RuntimeError("Incomplete grading output; not a negative task reward")
            report = get_eval_report(spec, {"instance_id": spec.instance_id, "model_patch": patch}, log, True)
            resolved = report[spec.instance_id]["resolved"]
            result = {
                "score": float(resolved),
                "resolved": resolved,
                "report": report,
                "executed_tests": True,
                "exit_code": test["returncode"],
            }
        result.update(sandbox_image=row["metadata"]["image_url"], sandbox_id=env.sandbox.sandbox_id)
        result.update(
            execution_image=getattr(env, "execution_image", None),
            registry_digest_before_create=getattr(env, "registry_digest", None),
        )
    write(output / "reward.json", result)
    return result
