"""Unit tests for the harness itself (no API key needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent


def test_sandbox_blocks_escape(tmp_path):
    (tmp_path / "ok.txt").write_text("hi")
    try:
        agent.tool_read(tmp_path, "../escape.txt")
    except ValueError as e:
        assert "escapes workspace" in str(e)
    else:
        raise AssertionError("should have blocked path traversal")


def test_edit_requires_unique_match(tmp_path):
    (tmp_path / "f.txt").write_text("a a a")
    assert "matches 3 times" in agent.tool_edit(tmp_path, "f.txt", "a", "b")
    assert "not found" in agent.tool_edit(tmp_path, "f.txt", "zzz", "b")


def test_truncation_and_bash(tmp_path):
    assert "truncated" in agent.truncate("x" * (agent.MAX_OUTPUT_CHARS + 5))
    r = agent.tool_bash(tmp_path, f"{sys.executable} -c \"print('hi')\"")
    assert "exit=0" in r and "hi" in r


KEY_VARS = ["OPENAI_API_KEY", "MODEL_API_KEY", "MUSE_SPARK_API_KEY", "META_API_KEY",
            "OPENAI_BASE_URL", "OPENAI_MODEL", "MUSE_SPARK_BASE_URL", "MUSE_SPARK_MODEL"]


def clear_keys(monkeypatch):
    for v in KEY_VARS:
        monkeypatch.delenv(v, raising=False)


def test_config_muse_defaults(monkeypatch):
    clear_keys(monkeypatch)
    monkeypatch.setenv("MODEL_API_KEY", "k")
    base, model, key = agent.llm_config()
    assert (base, model, key) == ("https://api.meta.ai/v1", "muse-spark-1.1", "k")


def test_config_openai_wins(monkeypatch):
    clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("MODEL_API_KEY", "m")
    base, model, key = agent.llm_config()
    assert (base, model, key) == ("https://api.openai.com/v1", "gpt-4o-mini", "o")


def test_config_missing_key(monkeypatch):
    clear_keys(monkeypatch)
    try:
        agent.llm_config()
    except RuntimeError as e:
        assert "MODEL_API_KEY" in str(e)
    else:
        raise AssertionError("should have raised without any key")


def test_write_file_guards(tmp_path):
    assert "ok: wrote" in agent.tool_write(tmp_path, "n.txt", "hi")
    assert "exists" in agent.tool_write(tmp_path, "n.txt", "hi")
    assert "ok:" in agent.tool_write(tmp_path, "n.txt", "bye", "true")
    assert "no_parent" in agent.tool_write(tmp_path, "nodir/n.txt", "hi")
    (tmp_path / "d").mkdir()
    assert "is_dir" in agent.tool_write(tmp_path, "d", "hi")
    assert "missing_arg" in agent.tool_write(tmp_path, "", "hi")


def test_dispatch_structured_errors(tmp_path):
    assert "unknown_tool" in agent.dispatch(tmp_path, "nope", {})
    assert "valid tools are" in agent.dispatch(tmp_path, "nope", {})
    assert "missing_arg" in agent.dispatch(tmp_path, "bash", {})
    assert "missing_arg" in agent.dispatch(tmp_path, "read_file", {})


def test_bash_blocked_and_missing(tmp_path):
    assert "blocked" in agent.tool_bash(tmp_path, "rm -rf / tmp")
    assert "missing_arg" in agent.tool_bash(tmp_path, "  ")


def _call(name, args, cid="c1"):
    import json as _j
    return {"content": "", "tool_calls": [{"id": cid, "function": {"name": name, "arguments": _j.dumps(args)}}]}


def test_run_stops_on_repeated_calls(tmp_path, monkeypatch):
    calls = [_call("bash", {"command": "echo hi"})] * 5
    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: calls.pop(0) if calls else _call("done", {"summary": "x"}))
    assert "stopped: no_progress" in agent.run("t", tmp_path, max_steps=6)


def test_run_stops_when_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: {"content": "thinking..."})
    assert "stopped: idle" in agent.run("t", tmp_path, max_steps=6)


def test_run_recovers_from_bad_json(tmp_path, monkeypatch):
    seq = [
        {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{not json"}}]},
        _call("done", {"summary": "recovered"}),
    ]
    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: seq.pop(0))
    assert agent.run("t", tmp_path, max_steps=4) == "recovered"


def test_read_paging(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(f"line{i}" for i in range(500)))
    first = agent.tool_read(tmp_path, "big.txt", 0, 100)
    assert "lines 1-100 of 500" in first
    assert "line0" in first
    assert "offset=100" in first
    second = agent.tool_read(tmp_path, "big.txt", 100, 100)
    assert "lines 101-200 of 500" in second
    assert "line100" in second
    assert "line99\n" not in second
    last = agent.tool_read(tmp_path, "big.txt", 498, 100)
    assert "lines 499-500 of 500" in last
    assert "offset=" not in last  # no continuation hint on the final page
    assert "past end" in agent.tool_read(tmp_path, "big.txt", 900, 100)
    assert "bad_args" in agent.tool_read(tmp_path, "big.txt", "xx", 10)
    assert "bad_args" in agent.tool_read(tmp_path, "big.txt", 0, "lots")
