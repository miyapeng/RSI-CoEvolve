"""OSWorld-Verified's PromptAgent / DesktopEnv / official single-example driver."""

from pathlib import Path
from types import SimpleNamespace

from baseline.records import read, sha

from .base import TaskInfo
from .official import OfficialBenchmark, OfficialTask, path_value, safe_id


class OSWorldVerified(OfficialBenchmark):
    name = "osworld_verified"
    url = "https://github.com/xlang-ai/OSWorld"
    commit = "8903ebeb0efdf74331bd7dc37ed91ccbbff22317"

    def discover(self, config):
        root = path_value(config["official"]["repo"]) / "evaluation_examples"
        suite = config["official"].get("suite", "test_all.json")
        if suite not in {"test_all.json", "test_nogdrive.json"}:
            raise ValueError("Choose an explicit official OSWorld suite")
        rows = []
        for domain, ids in read(root / suite).items():
            for iid in ids:
                task_id = safe_id(f"{domain}/{iid}")
                path = root / "examples" / f"{task_id}.json"
                task = read(path)
                if task["id"] != iid:
                    raise ValueError(f"OSWorld task ID mismatch: {path}")
                # evaluator/config files are trusted host inputs for DesktopEnv, not PromptAgent.
                rows.append(
                    OfficialTask(
                        TaskInfo(task_id, "xlang-ai/OSWorld-Verified", "final_test"),
                        {"task_id": task_id, "observations": [{"kind": "text", "text": task["instruction"]}]},
                        {"example_path": str(path), "suite": suite, "suite_sha256": sha(root / suite)},
                        sha(path),
                    )
                )
        return rows


def run_native(request):
    import os

    import lib_run_single
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.agent import PromptAgent

    o = request["options"]
    if o["provider_name"] not in {"docker", "vmware", "virtualbox"}:
        raise ValueError("This adapter currently validates local official VM providers only")
    if not o.get("path_to_vm") or not Path(o["path_to_vm"]).is_file():
        raise ValueError("Provide an existing official VM image; the adapter does not request image downloads")
    example = read(request["task"]["example_path"])
    root = Path(request["output"]) / "native"
    out = root / request["task_id"]
    out.mkdir(parents=True)
    # Official setup/evaluator modules resolve some supporting files relative to the checkout.
    os.chdir(request["source"]["repo"])
    agent = PromptAgent(
        model=request["model"],
        max_tokens=o["max_tokens"],
        top_p=o["top_p"],
        temperature=o["temperature"],
        action_space=o["action_space"],
        observation_type=o["observation_type"],
        max_trajectory_length=o["max_trajectory_length"],
    )
    env = DesktopEnv(
        provider_name=o["provider_name"],
        path_to_vm=o.get("path_to_vm"),
        action_space=agent.action_space,
        screen_size=(o["screen_width"], o["screen_height"]),
        headless=True,
        os_type="Ubuntu",
        cache_dir=str(root / "cache"),
        require_a11y_tree=o["observation_type"] in {"a11y_tree", "screenshot_a11y_tree", "som"},
    )
    scores = []
    args = SimpleNamespace(**{**o, "model": request["model"], "result_dir": str(root)})
    try:
        lib_run_single.run_single_example(
            agent, env, example, o["max_steps"], example["instruction"], args, str(out), scores
        )
    finally:
        env.close()
    # run_single_example calls env.evaluate() after interaction, using the original evaluator.
    if len(scores) != 1 or float((out / "result.txt").read_text()) != scores[0]:
        raise ValueError("Incomplete official OSWorld evaluation")
    return {
        "task_id": request["task_id"],
        "score": scores[0],
        "metrics": {"official_result": scores[0], "protocol": "PromptAgent/run_single_example"},
        "termination_reason": "official_evaluated",
    }
