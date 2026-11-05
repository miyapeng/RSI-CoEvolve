"""Replace only PromptAgent's model HTTP boundary, preserving its prompts/parser."""

import base64
import copy
import hashlib
import os
import time
from pathlib import Path

from baseline.records import write


def make_agent(options, model, output):
    from mm_agents.agent import PromptAgent
    from openai import OpenAI

    root = Path(output) / "model_calls"
    root.mkdir()
    client = OpenAI(
        api_key=os.environ[options["policy_api_key_env"]],
        base_url=options["policy_base_url"],
        timeout=options["model_request_timeout"],
        max_retries=0,
    )

    class GatewayAgent(PromptAgent):
        calls = 0
        usage = None
        returned_models = None
        last_error = None

        def call_llm(self, payload):
            self.calls += 1
            # Persist exact visible text with image file references, not base64 JSON.
            record = copy.deepcopy(payload)
            for message in record["messages"]:
                for part in message["content"]:
                    if part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        prefix = "data:image/png;base64,"
                        if not url.startswith(prefix):
                            raise ValueError("Expected the native PNG screenshot observation")
                        data = base64.b64decode(url[len(prefix) :], validate=True)
                        digest = hashlib.sha256(data).hexdigest()
                        image = root / f"{digest}.png"
                        image.write_bytes(data)
                        part["image_url"] = {"path": str(image), "sha256": digest, "detail": "high"}
            write(root / f"request_{self.calls:03d}.json", record)
            started = time.monotonic()
            try:
                response = client.chat.completions.create(**payload)
                text = response.choices[0].message.content
                if not text:
                    raise RuntimeError("Model returned no text for the official action parser")
            except Exception as exc:
                self.last_error = type(exc).__name__
                write(root / f"error_{self.calls:03d}.json", {"error_type": self.last_error})
                raise
            write(root / f"response_{self.calls:03d}.json", response.model_dump(mode="json"))
            write(root / f"timing_{self.calls:03d}.json", {"seconds": time.monotonic() - started})
            self.usage.append(response.usage.model_dump() if response.usage else None)
            self.returned_models.append(response.model)
            return text

    agent = GatewayAgent(
        model=model,
        max_tokens=options["max_tokens"],
        top_p=options["top_p"],
        temperature=options["temperature"],
        action_space=options["action_space"],
        observation_type=options["observation_type"],
        max_trajectory_length=options["max_trajectory_length"],
    )
    agent.usage, agent.returned_models = [], []
    return agent, client
