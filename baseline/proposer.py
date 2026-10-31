"""Explicit API protocol for the frozen external proposer; no policy training."""

import json
import os
from pathlib import Path

import httpx
from openai import OpenAI

from .records import write


def parse_json(content):
    """Accept bare JSON or one complete JSON fence, never arbitrary prose extraction."""
    text = content.strip()
    if text.startswith(("```json\n", "```\n")) and text.endswith("\n```"):
        text = text.split("\n", 1)[1][:-4]
    return json.loads(text)


def complete(settings, request, output):
    protocol = settings["protocol"]
    request = {**request, "model": os.environ[settings["model_env"]]}
    output = Path(output)
    key = os.environ[settings["key_env"]]
    with httpx.Client(timeout=settings["timeout"], trust_env=settings["trust_env"]) as http:
        if protocol == "anthropic_messages":
            # Claude's native endpoint does not accept Chat Completions' JSON mode.
            # The caller explicitly asks for JSON in the prompt and validates it.
            request = {k: request[k] for k in ("model", "messages", "max_tokens")}
            write(output / "proposal-input.json", {"protocol": protocol, "request": request})
            response = http.post(
                settings["base_url"].rstrip("/") + "/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json=request,
            )
            if not response.is_success:
                # Never serialize request headers or an untrusted gateway error body.
                write(output / "proposal-error.json", {"http_status": response.status_code})
                raise RuntimeError(f"Proposer Messages API returned HTTP {response.status_code}")
            body = response.json()
            write(output / "proposal-output.json", body)
            if body.get("stop_reason") != "end_turn":
                raise ValueError(f"Incomplete proposer response: {body.get('stop_reason')}")
            return "".join(b["text"] for b in body["content"] if b["type"] == "text")
        if protocol == "chat_completions":
            write(output / "proposal-input.json", {"protocol": protocol, "request": request})
            with OpenAI(base_url=settings["base_url"], api_key=key, max_retries=0, http_client=http) as client:
                response = client.chat.completions.create(**request)
            write(output / "proposal-output.json", response.model_dump(mode="json"))
            if response.choices[0].finish_reason != "stop":
                raise ValueError(f"Incomplete proposer response: {response.choices[0].finish_reason}")
            return response.choices[0].message.content
        raise ValueError("Unsupported proposer protocol")
