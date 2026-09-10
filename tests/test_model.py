"""Unit tests for model.py's provider strategy (no API key needed for the
translation logic; network is stubbed for the HTTP round-trip tests)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import model
import model_registry as mr
from model import (
    AnthropicProvider,
    OpenAIProvider,
    _from_anthropic_message,
    _to_anthropic_messages,
    resolve_provider,
)

TOOLS = [{"name": "read_file", "description": "read a file", "parameters": {"path": "file path"}}]

KEY_VARS = [
    "OPENAI_API_KEY", "MODEL_API_KEY", "MUSE_SPARK_API_KEY", "META_API_KEY",
    "OPENAI_BASE_URL", "OPENAI_MODEL", "MUSE_SPARK_BASE_URL", "MUSE_SPARK_MODEL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL", "ANTHROPIC_MAX_TOKENS",
]


def clear_keys(monkeypatch):
    for v in KEY_VARS:
        monkeypatch.delenv(v, raising=False)


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Never let a real ~/.config/harness/models.json leak into these tests."""
    monkeypatch.setenv("RIG_CONFIG_DIR", str(tmp_path))


def test_resolve_provider_prefers_openai(monkeypatch):
    clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    p = resolve_provider()
    assert isinstance(p, OpenAIProvider)


def test_resolve_provider_anthropic_when_no_openai_key(monkeypatch):
    clear_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    p = resolve_provider()
    assert isinstance(p, AnthropicProvider)
    assert p.model == model.ANTHROPIC_DEFAULT_MODEL
    assert p.base == model.ANTHROPIC_BASE_URL
    # Regression: a too-small max_tokens truncates the JSON of a tool call
    # that writes a long file (e.g. "generate test cases for this PR") before
    # the `content` argument is ever written, so the model retries with the
    # same incomplete call until the repeat-call guard gives up.
    assert p.max_tokens == model.ANTHROPIC_DEFAULT_MAX_TOKENS
    assert p.max_tokens >= 8192


def test_resolve_provider_anthropic_max_tokens_env_override(monkeypatch):
    clear_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "16000")
    p = resolve_provider()
    assert p.max_tokens == 16000


def test_resolve_provider_anthropic_env_overrides(monkeypatch):
    clear_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-x")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.test/v1")
    p = resolve_provider()
    assert p.model == "claude-opus-x"
    assert p.base == "https://example.test/v1"


def test_resolve_provider_falls_back_to_muse(monkeypatch):
    clear_keys(monkeypatch)
    monkeypatch.setenv("MODEL_API_KEY", "k")
    p = resolve_provider()
    assert isinstance(p, OpenAIProvider)
    assert p.model == "muse-spark-1.1"


def test_resolve_provider_raises_when_nothing_configured(monkeypatch):
    clear_keys(monkeypatch)
    with pytest.raises(RuntimeError):
        resolve_provider()


def test_resolve_provider_falls_back_to_registry_when_no_env_vars(monkeypatch):
    clear_keys(monkeypatch)
    mr.add_group("work", "anthropic", "sk-ant-xyz", "claude-sonnet-5")
    p = resolve_provider()
    assert isinstance(p, AnthropicProvider)
    assert p.model == "claude-sonnet-5"
    assert p.key == "sk-ant-xyz"


def test_resolve_provider_env_vars_win_over_registry(monkeypatch):
    clear_keys(monkeypatch)
    mr.add_group("work", "anthropic", "sk-ant-xyz", "claude-sonnet-5")
    monkeypatch.setenv("MODEL_API_KEY", "k")
    p = resolve_provider()
    assert isinstance(p, OpenAIProvider)
    assert p.model == "muse-spark-1.1"


def test_resolve_provider_registry_openai_profile(monkeypatch):
    clear_keys(monkeypatch)
    mr.add_group("cheap", "openai", "sk-oai", "gpt-4o-mini")
    p = resolve_provider()
    assert isinstance(p, OpenAIProvider)
    assert p.model == "gpt-4o-mini"
    assert p.key == "sk-oai"


def test_to_anthropic_messages_extracts_system_and_merges_tool_results():
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "do the thing"},
        {"content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
            {"id": "c2", "function": {"name": "read_file", "arguments": '{"path": "b.txt"}'}},
        ], "role": "assistant"},
        {"role": "tool", "tool_call_id": "c1", "content": "contents of a"},
        {"role": "tool", "tool_call_id": "c2", "content": "contents of b"},
    ]
    system, out = _to_anthropic_messages(messages)
    assert system == "be nice"
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    # both tool results merge into a single user turn (Anthropic requires alternation)
    result_turn = out[2]
    assert len(result_turn["content"]) == 2
    assert all(b["type"] == "tool_result" for b in result_turn["content"])
    assert {b["tool_use_id"] for b in result_turn["content"]} == {"c1", "c2"}
    # assistant turn carries both tool_use blocks with parsed input
    assistant_turn = out[1]
    tool_uses = [b for b in assistant_turn["content"] if b["type"] == "tool_use"]
    assert len(tool_uses) == 2
    assert tool_uses[0]["input"] == {"path": "a.txt"}


def test_from_anthropic_message_translates_text_and_tool_use():
    raw = {"content": [
        {"type": "text", "text": "let me check"},
        {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.txt"}},
    ]}
    msg = _from_anthropic_message(raw)
    assert msg["content"] == "let me check"
    assert msg["tool_calls"] == [
        {"id": "t1", "function": {"name": "read_file", "arguments": json.dumps({"path": "a.txt"})}}
    ]


def test_from_anthropic_message_text_only_has_no_tool_calls():
    msg = _from_anthropic_message({"content": [{"type": "text", "text": "hi"}]})
    assert msg["content"] == "hi"
    assert "tool_calls" not in msg


class _FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_anthropic_provider_call_round_trip(monkeypatch):
    import urllib.request as _urllib_req

    captured = {}

    def fake_urlopen(req, timeout=120):
        captured["url"] = req.full_url
        captured["headers"] = req.headers
        captured["body"] = json.loads(req.data)
        return _FakeResp({"content": [{"type": "text", "text": "done"}]})

    monkeypatch.setattr(_urllib_req, "urlopen", fake_urlopen)
    provider = AnthropicProvider(model="claude-x", key="sk-ant-test")
    msg = provider.call([{"role": "user", "content": "hi"}], TOOLS)

    assert msg == {"role": "assistant", "content": "done"}
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["X-api-key"] == "sk-ant-test"
    assert captured["headers"]["Anthropic-version"] == model.ANTHROPIC_VERSION
    assert captured["body"]["model"] == "claude-x"
    assert captured["body"]["tools"][0]["name"] == "read_file"
    assert captured["body"]["tools"][0]["input_schema"]["properties"]["path"]["description"] == "file path"
    # Default provider (no explicit max_tokens) sends the raised default, not
    # the old 4096 that truncated large tool-call arguments mid-JSON.
    assert captured["body"]["max_tokens"] == model.ANTHROPIC_DEFAULT_MAX_TOKENS
