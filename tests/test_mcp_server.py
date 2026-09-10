"""Unit tests for mcp_server.py — Rig exposed as an MCP server."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
import mcp_server


def test_silent_renderer_denies_risky_bash_without_running_it(tmp_path, monkeypatch):
    """The whole point: no human on an MCP call, so a risk="confirm" bash
    call must be refused, not silently run or hung waiting for an answer."""
    import json as _j

    calls = [
        {"content": "", "tool_calls": [{"id": "c1", "function": {
            "name": "bash", "arguments": _j.dumps({"command": "rm -rf build", "risk": "confirm"})}}]},
        {"content": "", "tool_calls": [{"id": "c2", "function": {
            "name": "done", "arguments": _j.dumps({"summary": "gave up"})}}]},
    ]
    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: calls.pop(0))
    dispatch_calls = []
    monkeypatch.setattr(agent, "dispatch", lambda *a, **k: dispatch_calls.append(a) or "ok")

    result = mcp_server.delegate_task(tmp_path, "clean up", max_steps=3)

    assert result == "gave up"
    assert dispatch_calls == []


def test_silent_renderer_runs_safe_bash_unprompted(tmp_path, monkeypatch):
    """auto mode: calls without risk="confirm" run without any gate at all."""
    import json as _j

    calls = [
        {"content": "", "tool_calls": [{"id": "c1", "function": {
            "name": "bash", "arguments": _j.dumps({"command": "pytest -q"})}}]},
        {"content": "", "tool_calls": [{"id": "c2", "function": {
            "name": "done", "arguments": _j.dumps({"summary": "tests pass"})}}]},
    ]
    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: calls.pop(0))
    dispatch_calls = []
    monkeypatch.setattr(agent, "dispatch", lambda *a, **k: dispatch_calls.append(a) or "ok")

    result = mcp_server.delegate_task(tmp_path, "run the tests", max_steps=3)

    assert result == "tests pass"
    assert len(dispatch_calls) == 1


def test_delegate_task_passes_workspace_and_max_steps_through(tmp_path, monkeypatch):
    captured = {}

    def fake_run(task, root, max_steps=agent.DEFAULT_MAX_STEPS, log_path=None, renderer=None, **kw):
        captured.update(task=task, root=root, max_steps=max_steps, renderer=renderer)
        return "delegated ok"

    monkeypatch.setattr(agent, "run", fake_run)

    result = mcp_server.delegate_task(tmp_path, "fix the typo", max_steps=7)

    assert result == "delegated ok"
    assert captured["task"] == "fix the typo"
    assert captured["root"] == tmp_path
    assert captured["max_steps"] == 7
    assert captured["renderer"].approval_mode == "auto"
    assert captured["renderer"].confirm_bash("anything") is False


def test_build_server_registers_delegate_task_tool(tmp_path, monkeypatch):
    server = mcp_server.build_server(tmp_path)

    tools = asyncio.run(server.list_tools())
    names = [t.name for t in tools]
    assert "delegate_task_tool" in names

    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: {
        "content": "", "tool_calls": [{"id": "c1", "function": {
            "name": "done", "arguments": '{"summary": "hi from rig"}'}}],
    })
    outcome = asyncio.run(server.call_tool("delegate_task_tool", {"task": "say hi"}))
    # FastMCP wraps sync-tool results as content blocks; find the text somewhere in it.
    assert "hi from rig" in str(outcome)
