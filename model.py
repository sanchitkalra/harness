"""Model client: provider strategies + one chat-completions call.

Knows nothing about tools, files, or terminal output. Takes message
dicts and tool schemas, returns the model's parsed message dict in the
OpenAI wire shape: {"content": str, "tool_calls": [{"id", "function":
{"name", "arguments"}}]}. Every Provider translates to/from that shape,
so agent.py never needs to know which provider is active.

Add a new provider: write a class with a `model` attribute and a
`call(messages, tools) -> dict` method (see OpenAIProvider /
AnthropicProvider), then wire it into resolve_provider().
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

META_BASE_URL = "https://api.meta.ai/v1"
META_DEFAULT_MODEL = "muse-spark-1.1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-5"
ANTHROPIC_VERSION = "2023-06-01"

# Network retry budget (seconds of total backoff sleep). Delays grow
# exponentially from RETRY_BASE_S, capped at RETRY_CAP_S each.
RETRY_BUDGET_S = 30.0
RETRY_BASE_S = 1.0
RETRY_CAP_S = 8.0


class ApiError(RuntimeError):
    """Non-retryable API error (4xx except 429) with status, reason, snippet."""

    def __init__(self, status_code: int, reason: str, body_snippet: str):
        self.status_code = status_code
        self.reason = reason
        self.body_snippet = body_snippet
        super().__init__(f"API error {status_code} {reason}: {body_snippet}")


class NetworkError(RuntimeError):
    """The LLM HTTP call failed even after retrying within budget."""


def _retryable(exc: Exception) -> bool:
    """Only transient failures retry: timeouts, DNS/dropped connections,
    HTTP 429 and 5xx. Other 4xx (auth, bad request) fail immediately —
    retrying those just burns the budget."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600
    return isinstance(exc, (urllib.error.URLError, OSError))


def _call_with_retries(req: urllib.request.Request) -> dict:
    """POST req, retrying transient failures with exponential backoff
    within RETRY_BUDGET_S. Returns the parsed JSON body."""
    delay = RETRY_BASE_S
    waited = 0.0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            if not _retryable(e):
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        raw = e.read()
                        txt = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw or "")
                    except Exception:
                        txt = ""
                    reason = e.reason if isinstance(e.reason, str) else str(e.reason)
                    raise ApiError(e.code, reason, txt[:500]) from e
                raise
            if waited + delay > RETRY_BUDGET_S:
                raise NetworkError(f"llm call failed after retries (~{waited:.0f}s): {e}") from e
            time.sleep(delay)
            waited += delay
            delay = min(delay * 2, RETRY_CAP_S)


def llm_config() -> tuple[str, str, str]:
    """Resolve (base_url, model, key) for an OpenAI-compatible provider from
    env. OpenAI vars win over Muse Spark when both are set."""
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
    raise RuntimeError("set MODEL_API_KEY (Muse Spark), OPENAI_API_KEY, or ANTHROPIC_API_KEY")


# ----------------------------------------------------------------------
# Provider strategy
# ----------------------------------------------------------------------

class Provider(Protocol):
    model: str

    def call(self, messages: list[dict], tools: list[dict]) -> dict: ...


def _openai_tool_schema(tools: list[dict]) -> list[dict]:
    return [
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


@dataclass
class OpenAIProvider:
    """OpenAI-compatible chat-completions API (OpenAI itself, Muse Spark, and
    any other provider that speaks the same /chat/completions wire format)."""

    base: str
    model: str
    key: str

    def call(self, messages: list[dict], tools: list[dict]) -> dict:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "tools": _openai_tool_schema(tools),
        }).encode()
        req = urllib.request.Request(
            f"{self.base.rstrip('/')}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        raw = _call_with_retries(req)
        return raw["choices"][0]["message"]


def _anthropic_tool_schema(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in t["parameters"].items()},
            },
        }
        for t in tools
    ]


def _to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Translate OpenAI-shaped messages into (system_prompt, anthropic_messages).

    Anthropic requires strictly alternating user/assistant turns, so
    consecutive tool-result / user messages (our drive() loop emits one
    "tool" message per tool call before the next assistant turn) are
    merged into a single user turn with multiple content blocks.
    """
    system_parts: list[str] = []
    out: list[dict] = []

    def push_user_block(block: dict) -> None:
        if out and out[-1]["role"] == "user":
            out[-1]["content"].append(block)
        else:
            out.append({"role": "user", "content": [block]})

    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content"):
                system_parts.append(str(m["content"]))
        elif role == "user":
            push_user_block({"type": "text", "text": m.get("content") or ""})
        elif role == "tool":
            push_user_block({
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": m.get("content") or "",
            })
        elif role == "assistant":
            blocks: list[dict] = []
            content = (m.get("content") or "").strip()
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in m.get("tool_calls") or []:
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except Exception:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": args,
                })
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        # unknown roles are dropped — nothing today emits any.
    return "\n".join(system_parts), out


def _from_anthropic_message(raw: dict) -> dict:
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in raw.get("content") or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })
    msg: dict = {"role": "assistant", "content": "\n".join(t for t in text_parts if t)}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


@dataclass
class AnthropicProvider:
    """Anthropic Messages API (https://api.anthropic.com/v1/messages)."""

    base: str = ANTHROPIC_BASE_URL
    model: str = ANTHROPIC_DEFAULT_MODEL
    key: str = ""
    max_tokens: int = 4096

    def call(self, messages: list[dict], tools: list[dict]) -> dict:
        system, anthropic_messages = _to_anthropic_messages(messages)
        body_obj: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
            "tools": _anthropic_tool_schema(tools),
        }
        if system:
            body_obj["system"] = system
        req = urllib.request.Request(
            f"{self.base.rstrip('/')}/messages",
            data=json.dumps(body_obj).encode(),
            headers={
                "x-api-key": self.key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
        )
        raw = _call_with_retries(req)
        return _from_anthropic_message(raw)


def _provider_from_profile(profile: dict) -> Provider:
    provider = profile.get("provider")
    key = profile.get("api_key", "")
    base = profile.get("base_url") or None
    if provider == "anthropic":
        return AnthropicProvider(base=base or ANTHROPIC_BASE_URL, model=profile.get("model") or ANTHROPIC_DEFAULT_MODEL, key=key)
    return OpenAIProvider(base=base or "https://api.openai.com/v1", model=profile.get("model") or "gpt-4o-mini", key=key)


def resolve_provider() -> Provider:
    """Pick a Provider strategy: env vars (OpenAI > Anthropic > Muse Spark)
    first, then the persisted registry's active profile (see
    model_registry.py — set via /model in the TUI, or by hand-editing
    ~/.config/harness/models.json).

    Raises RuntimeError if nothing is configured either way.
    """
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider(
            base=os.environ.get("ANTHROPIC_BASE_URL", ANTHROPIC_BASE_URL),
            model=os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_DEFAULT_MODEL),
            key=os.environ["ANTHROPIC_API_KEY"],
        )
    try:
        base, model, key = llm_config()
        return OpenAIProvider(base, model, key)
    except RuntimeError:
        import model_registry
        profile = model_registry.get_active_profile()
        if profile is None:
            raise
        return _provider_from_profile(profile)


def llm_call(messages: list[dict], tools: list[dict]) -> dict:
    """One model call, routed to whichever provider is configured.

    Transient network failures retry with exponential backoff within
    RETRY_BUDGET_S, then raise NetworkError. Non-retryable errors
    (bad key, bad request, unparseable reply) raise as before.
    """
    return resolve_provider().call(messages, tools)
