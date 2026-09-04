"""Model client: provider config + one chat-completions call.

Knows nothing about tools, files, or terminal output. Takes message
dicts and tool schemas, returns the model's parsed message dict.
"""
from __future__ import annotations

import json
import os
import urllib.request

META_BASE_URL = "https://api.meta.ai/v1"
META_DEFAULT_MODEL = "muse-spark-1.1"


def llm_config() -> tuple[str, str, str]:
    """Resolve (base_url, model, key) from env. OpenAI vars win when set."""
    if "OPENAI_API_KEY" in os.environ:
        return (
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            os.environ["OPENAI_API_KEY"],
        )
    for var in ("MODEL_API_KEY", "MUSE_SPARK_API_KEY", "META_API_KEY"):
        if os.environ.get(var):
            return (
                os.environ.get("MUSE_SPARK_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL", META_BASE_URL),
                os.environ.get("MUSE_SPARK_MODEL")
                or os.environ.get("OPENAI_MODEL", META_DEFAULT_MODEL),
                os.environ[var],
            )
    raise RuntimeError("set MODEL_API_KEY (Muse Spark) or OPENAI_API_KEY")


def llm_call(messages: list[dict], tools: list[dict]) -> dict:
    """One chat-completions call (OpenAI-compatible). Returns parsed message."""
    base, model, key = llm_config()
    schema = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in t["parameters"]},
                },
            },
        }
        for t in tools
    ]
    body = json.dumps({"model": model, "messages": messages, "tools": schema}).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return data["choices"][0]["message"]
