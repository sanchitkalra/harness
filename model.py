"""Model client: provider config + one chat-completions call.

Knows nothing about tools, files, or terminal output. Takes message
dicts and tool schemas, returns the model's parsed message dict.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

META_BASE_URL = "https://api.meta.ai/v1"
META_DEFAULT_MODEL = "muse-spark-1.1"

# Network retry budget (seconds of total backoff sleep). Delays grow
# exponentially from RETRY_BASE_S, capped at RETRY_CAP_S each.
RETRY_BUDGET_S = 30.0
RETRY_BASE_S = 1.0
RETRY_CAP_S = 8.0


class NetworkError(RuntimeError):
    """The LLM HTTP call failed even after retrying within budget."""


def _retryable(exc: Exception) -> bool:
    """Only transient failures retry: timeouts, DNS/dropped connections,
    HTTP 429 and 5xx. Other 4xx (auth, bad request) fail immediately —
    retrying those just burns the budget."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600
    return isinstance(exc, (urllib.error.URLError, OSError))


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
    """One chat-completions call (OpenAI-compatible). Returns parsed message.

    Transient network failures retry with exponential backoff within
    RETRY_BUDGET_S, then raise NetworkError. Non-retryable errors
    (bad key, bad request, unparseable reply) raise as before.
    """
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
    delay = RETRY_BASE_S
    waited = 0.0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            return data["choices"][0]["message"]
        except Exception as e:
            if not _retryable(e) or waited + delay > RETRY_BUDGET_S:
                if _retryable(e):
                    raise NetworkError(f"llm call failed after retries (~{waited:.0f}s): {e}") from e
                raise
            time.sleep(delay)
            waited += delay
            delay = min(delay * 2, RETRY_CAP_S)
