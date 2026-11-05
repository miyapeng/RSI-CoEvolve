"""OSWorld-Verified's PromptAgent / DesktopEnv / official single-example driver."""

from pathlib import Path
from types import SimpleNamespace

from baseline.records import read, sha, write

from .base import TaskInfo
from .official import OfficialBenchmark, OfficialTask, path_value, safe_id


class OSWorldVerified(OfficialBenchmark):
    name = "osworld_verified"
    url = "https://github.com/xlang-ai/OSWorld"
    commit = "8903ebeb0efdf74331bd7dc37ed91ccbbff22317"

    def plan(self, task, checkpoint, harness, config, output, seed):
        options = dict(config["official"])
        if options["provider_name"] == "tencent_e2b":
            from .osworld_tencent import source

            options["tencent_source"] = source(options)
        return super().plan(task, checkpoint, harness, {**config, "official": options}, output, seed)

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
    import contextlib
    import os

    import lib_run_single
    from mm_agents.agent import PromptAgent

    o = request["options"]
    tencent = o["provider_name"] == "tencent_e2b"
    if o["provider_name"] not in {"docker", "vmware", "virtualbox", "tencent_e2b"}:
        raise ValueError("Unsupported OSWorld provider")
    if not tencent and (not o.get("path_to_vm") or not Path(o["path_to_vm"]).is_file()):
        raise ValueError("Provide an existing official VM image; the adapter does not request image downloads")
    example = read(request["task"]["example_path"])
    root = Path(request["output"]) / "native"
    out = root / request["task_id"]
    out.mkdir(parents=True)
    # Official setup/evaluator modules resolve some supporting files relative to the checkout.
    os.chdir(request["source"]["repo"])
    extra = {}
    with contextlib.ExitStack() as stack:
        if o.get("policy_transport") == "openai_chat":
            from .osworld_gateway import make_agent

            agent, client = make_agent(o, request["model"], request["output"])
            stack.callback(client.close)
        else:
            agent = PromptAgent(
                model=request["model"],
                max_tokens=o["max_tokens"],
                top_p=o["top_p"],
                temperature=o["temperature"],
                action_space=o["action_space"],
                observation_type=o["observation_type"],
                max_trajectory_length=o["max_trajectory_length"],
            )
        if tencent:
            from .osworld_tencent import prepare_environment, provider_pair

            DesktopEnv = prepare_environment(request)
            manager, provider = provider_pair(o, request["output"])
            extra = {"provider_pair": (manager, provider)}
            # Also covers a failure during the environment constructor.
            stack.callback(provider.stop_emulator, None)
        else:
            from desktop_env.desktop_env import DesktopEnv
        env = DesktopEnv(
            provider_name=o["provider_name"],
            path_to_vm=o["tencent"]["template"] if tencent else o.get("path_to_vm"),
            action_space=agent.action_space,
            screen_size=(o["screen_width"], o["screen_height"]),
            headless=True,
            os_type="Ubuntu",
            cache_dir=str(root / "cache"),
            require_a11y_tree=o["observation_type"] in {"a11y_tree", "screenshot_a11y_tree", "som"},
            **extra,
        )
        stack.callback(env.close)
        if tencent:
            from io import BytesIO

            from PIL import Image

            screenshot = env.controller.get_screenshot()
            if not screenshot:
                raise RuntimeError("No valid screenshot returned by the remote desktop")
            if Image.open(BytesIO(screenshot)).size != (o["screen_width"], o["screen_height"]):
                raise ValueError("Remote desktop resolution differs from the recorded protocol")
            (out / "initial_desktop.png").write_bytes(screenshot)
            write(
                out / "environment_setup.json",
                {
                    "recording_server": env.recording_server_info,
                    "keyboard_autorepeat": env.key_autorepeat_status,
                    "template": o["tencent"]["template"],
                    "tool_id": o["tencent"]["tool_id"],
                },
            )
        scores = []
        args = SimpleNamespace(**{**o, "model": request["model"], "result_dir": str(root)})
        lib_run_single.run_single_example(
            agent, env, example, o["max_steps"], example["instruction"], args, str(out), scores
        )
        if getattr(agent, "last_error", None):
            raise RuntimeError("OSWorld model call failed: " + agent.last_error)
    # run_single_example calls env.evaluate() after interaction, using the original evaluator.
    if len(scores) != 1 or float((out / "result.txt").read_text()) != scores[0]:
        raise ValueError("Incomplete official OSWorld evaluation")
    return {
        "task_id": request["task_id"],
        "score": scores[0],
        "metrics": {
            "official_result": scores[0],
            "protocol": "PromptAgent/run_single_example",
            "model_calls": getattr(agent, "calls", None),
            "usage": getattr(agent, "usage", None),
            "returned_models": getattr(agent, "returned_models", None),
            "provider": o["provider_name"],
        },
        "termination_reason": "official_evaluated",
    }
