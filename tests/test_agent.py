"""Unit tests for the harness itself (no API key needed)."""
import sys
from pathlib import Path

import pytest

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

def test_session_header_roundtrip(tmp_path, monkeypatch):
    import json
    # ensure llm_config returns a known model
    monkeypatch.setenv("MODEL_API_KEY", "k")
    # remove openai key if set to get deterministic model
    for v in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "MUSE_SPARK_BASE_URL", "MUSE_SPARK_MODEL"]:
        monkeypatch.delenv(v, raising=False) if v != "MODEL_API_KEY" else None
        # Actually ensure OPENAI_API_KEY cleared
        if v == "OPENAI_API_KEY":
            monkeypatch.delenv(v, raising=False)
    # monkeypatch llm_call to finish immediately
    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: _call("done", {"summary": "header ok"}))
    log_path = tmp_path / "sessions" / "20240101-000000.jsonl"
    result = agent.run("task for header test", tmp_path, max_steps=3, log_path=log_path)
    assert result == "header ok"
    assert log_path.exists()
    header = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert header["type"] == "session"
    assert header["workspace"] == str(tmp_path)
    assert "model" in header and header["model"]
    # check model value when env gives default muse model
    assert header["model"] == "muse-spark-1.1"

    # now test fallback to unknown when llm_config fails
    def fail_config():
        raise RuntimeError("no key")
    monkeypatch.setattr(agent, "llm_config", fail_config)
    log_path2 = tmp_path / "sessions" / "20240101-000001.jsonl"
    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: _call("done", {"summary": "ok2"}))
    agent.run("another task", tmp_path, max_steps=3, log_path=log_path2)
    header2 = json.loads(log_path2.read_text(encoding="utf-8").splitlines()[0])
    assert header2["model"] == "unknown"
    assert header2["workspace"] == str(tmp_path)


def test_load_instructions_precedence_and_truncation(tmp_path):
    # empty -> ""
    assert agent.load_instructions(tmp_path) == ""
    # CLAUDE.md fallback
    (tmp_path / "CLAUDE.md").write_text("from claude")
    assert agent.load_instructions(tmp_path) == "from claude"
    # AGENTS.md wins over CLAUDE.md
    (tmp_path / "AGENTS.md").write_text("from agents")
    assert agent.load_instructions(tmp_path) == "from agents"
    # truncates to 2000 chars
    (tmp_path / "AGENTS.md").write_text("x" * 5000)
    assert len(agent.load_instructions(tmp_path)) == 2000


def test_run_injects_instructions(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("INJECTED_GUIDE")
    captured = {}

    def fake_llm_call(messages, tools):
        captured["messages"] = messages
        return _call("done", {"summary": "ok"})

    monkeypatch.setattr(agent, "llm_call", fake_llm_call)
    agent.run("do thing", tmp_path, max_steps=2)
    sys_msg = captured["messages"][0]["content"]
    assert "INJECTED_GUIDE" in sys_msg
    assert agent.SYSTEM in sys_msg


def test_sessions_dir_and_list(tmp_path, capsys):
    sdir = agent.sessions_dir(tmp_path)
    assert sdir == tmp_path / "sessions"
    assert sdir.exists()
    # create a dummy session file
    (sdir / "a.jsonl").write_text('{"type":"session","id":"a","ts":"2020-01-01T00:00:00+00:00","task":"hello"}\n')
    (sdir / "bad.jsonl").write_text('not json\n')
    agent.list_sessions(tmp_path)
    out = capsys.readouterr().out
    assert "a 2020-01-01T00:00:00+00:00 hello" in out
    assert "bad (no header)" in out


def test_fork_parent_id_in_header(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: _call("done", {"summary": "child ok"}))
    sdir = agent.sessions_dir(tmp_path)
    # create parent session with 2 extra steps
    parent_id = "parent123"
    (sdir / f"{parent_id}.jsonl").write_text(
        '{"type":"session","id":"parent123","ts":"2020-01-01T00:00:00+00:00","task":"parent task"}\n'
        '{"step":1,"tool":"bash"}\n'
        '{"step":2,"tool":"write_file"}\n'
    )
    child_path = sdir / "child.jsonl"
    result = agent.run("new task", tmp_path, max_steps=2, log_path=child_path, parent_id=parent_id)
    assert result == "child ok"
    header = json.loads(child_path.read_text(encoding="utf-8").splitlines()[0])
    assert header["parent_id"] == parent_id
    assert header["task"] == "new task"


def test_main_fork_cli(tmp_path, monkeypatch, capsys):
    import json
    sdir = agent.sessions_dir(tmp_path)
    parent_id = "20200101-000000"
    (sdir / f"{parent_id}.jsonl").write_text(
        '{"type":"session","id":"20200101-000000","ts":"2020-01-01T00:00:00+00:00","task":"do parent thing"}\n'
        '{"step":1}\n{"step":2}\n'
    )
    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: _call("done", {"summary": "forked ok"}))
    monkeypatch.setattr(agent, "llm_config", lambda: ("https://api.meta.ai/v1", "muse-spark-1.1", "k"))
    monkeypatch.setattr("sys.argv", ["agent", "--workspace", str(tmp_path), "--fork", parent_id, "continue work"])
    agent.main()
    # find child session (different from parent)
    files = [p for p in sdir.glob("*.jsonl") if p.stem != parent_id]
    assert len(files) == 1
    header = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert header["parent_id"] == parent_id
    assert "Forked from session" in header["task"]
    assert parent_id in header["task"]
    assert "do parent thing" in header["task"]
    assert "steps=2" in header["task"] or "step count" in header["task"].lower() or "2" in header["task"]
    assert "continue work" in header["task"]

def test_web_search_success_and_truncation(tmp_path, monkeypatch):
    import json as _json
    import urllib.request as _urllib_req

    payload = {
        "query": {
            "search": [
                {"title": "Python (programming language)", "snippet": 'Python is <span class="searchmatch">great</span>'},
                {"title": "Monty Python", "snippet": "comedy group"},
            ]
        }
    }
    raw = _json.dumps(payload).encode()

    class FakeResp:
        def __init__(self, data):
            self._data = data
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=15):
        # check User-Agent present
        assert req.get_header("User-agent") or req.headers.get("User-Agent")
        return FakeResp(raw)

    monkeypatch.setattr(_urllib_req, "urlopen", fake_urlopen)

    out = agent.tool_search(tmp_path, "python", 2)
    assert "Python (programming language)" in out
    assert "Monty Python" in out
    assert "great" in out  # html stripped
    assert "https://en.wikipedia.org/wiki/" in out
    assert "<span" not in out


def test_web_search_network_failure(tmp_path, monkeypatch):
    import time as _time
    import urllib.request as _urllib_req

    def fake_urlopen_fail(req, timeout=15):
        raise OSError("network down")

    monkeypatch.setattr(_urllib_req, "urlopen", fake_urlopen_fail)
    monkeypatch.setattr(_time, "sleep", lambda s: None)  # retry budget elapses instantly
    out = agent.tool_search(tmp_path, "python", 5)
    assert out.startswith("error:")
    assert "network" in out.lower() or "failure" in out.lower()


def test_web_search_dispatch_bad_args(tmp_path):
    # missing query arg -> error via dispatch
    assert "missing_arg" in agent.dispatch(tmp_path, "web_search", {})
    assert "missing_arg" in agent.dispatch(tmp_path, "web_search", {"query": "   "})
    # bad limit
    assert "bad_args" in agent.dispatch(tmp_path, "web_search", {"query": "hi", "limit": "notanint"})


class _FakeResp:
    """Minimal urlopen context manager yielding canned JSON bytes."""

    def __init__(self, payload):
        import json as _j
        self._data = _j.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_sleep(monkeypatch):
    import time as _time
    slept = []
    monkeypatch.setattr(_time, "sleep", slept.append)
    return slept


def _llm_key(monkeypatch):
    clear_keys(monkeypatch)
    monkeypatch.setenv("MODEL_API_KEY", "k")


def test_llm_call_retries_transient_then_succeeds(monkeypatch):
    import urllib.request as _urllib_req

    _llm_key(monkeypatch)
    attempts = []

    def flaky(req, timeout=120):
        attempts.append(1)
        if len(attempts) <= 2:
            raise OSError("network down")
        return _FakeResp({"choices": [{"message": {"role": "assistant", "content": "hi"}}]})

    monkeypatch.setattr(_urllib_req, "urlopen", flaky)
    slept = _stub_sleep(monkeypatch)
    msg = agent.llm_call([{"role": "user", "content": "hi"}], [])
    assert msg == {"role": "assistant", "content": "hi"}
    assert slept == [1.0, 2.0]  # exponential backoff from the 1s base


def test_llm_call_gives_up_within_budget(monkeypatch):
    import urllib.request as _urllib_req

    _llm_key(monkeypatch)

    def always_down(req, timeout=120):
        raise OSError("network down")

    monkeypatch.setattr(_urllib_req, "urlopen", always_down)
    slept = _stub_sleep(monkeypatch)
    with pytest.raises(agent.NetworkError, match="after retries"):
        agent.llm_call([{"role": "user", "content": "hi"}], [])
    assert slept, "should have backed off at least once"
    assert sum(slept) <= 30.0, f"backoff exceeded 30s bound: {slept}"
    assert all(b >= a for a, b in zip(slept, slept[1:])), f"backoff should grow: {slept}"


def test_llm_call_no_retry_on_client_error(monkeypatch):
    import urllib.error as _urlerr
    import urllib.request as _urllib_req

    _llm_key(monkeypatch)

    def forbidden(req, timeout=120):
        raise _urlerr.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(_urllib_req, "urlopen", forbidden)
    slept = _stub_sleep(monkeypatch)
    with pytest.raises(agent.ApiError):
        agent.llm_call([{"role": "user", "content": "hi"}], [])
    assert slept == [], "4xx must fail fast without sleeping"


def test_llm_call_retries_server_error(monkeypatch):
    import urllib.error as _urlerr
    import urllib.request as _urllib_req

    _llm_key(monkeypatch)

    def unavailable(req, timeout=120):
        raise _urlerr.HTTPError(req.full_url, 503, "Unavailable", {}, None)

    monkeypatch.setattr(_urllib_req, "urlopen", unavailable)
    slept = _stub_sleep(monkeypatch)
    with pytest.raises(agent.NetworkError, match="after retries"):
        agent.llm_call([{"role": "user", "content": "hi"}], [])
    assert slept and sum(slept) <= 30.0


def test_run_returns_stopped_on_network_failure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise agent.NetworkError("llm call failed after retries (~23s): down")

    monkeypatch.setattr(agent, "llm_call", boom)
    out = agent.run("t", tmp_path, max_steps=3)
    assert out.startswith("stopped: network")


def test_web_search_retries_then_succeeds(tmp_path, monkeypatch):
    import urllib.request as _urllib_req

    payload = {"query": {"search": [{"title": "Python", "snippet": "great"}]}}
    attempts = []

    def flaky(req, timeout=15):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("network down")
        return _FakeResp(payload)

    monkeypatch.setattr(_urllib_req, "urlopen", flaky)
    slept = _stub_sleep(monkeypatch)
    out = agent.tool_search(tmp_path, "python", 5)
    assert "Python" in out
    assert slept == [1.0]


def test_ui_read_summary_no_color(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    import ui
    ui.tool_result("read_file", "lines 1-2 of 10\nfoo\nbar\n... [more]")
    out = capsys.readouterr().out
    assert "lines 1-2 of 10" in out
    # header-only: body must not appear
    assert "foo" not in out
    assert "bar" not in out


def test_ui_edit_diff_markers_no_color(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    import ui
    diff_text = "ok: edited foo.txt\n--- a/foo.txt\n+++ b/foo.txt\n@@ -1 +1 @@\n-old line\n+new line\n context"
    ui.tool_result("edit_file", diff_text)
    out = capsys.readouterr().out
    assert "ok: edited foo.txt" in out
    assert "-old line" in out
    assert "+new line" in out
    assert "---" in out or "+++" in out or "@@" in out


def test_ui_write_diff_markers_no_color(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    import ui
    write_text = "ok: wrote new.txt (12 chars)\n+hello world\n+second line"
    ui.tool_result("write_file", write_text)
    out = capsys.readouterr().out
    assert "ok: wrote new.txt" in out
    assert "+hello world" in out
    assert "+second line" in out


def test_ui_error_passthrough_no_color(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    import ui
    for name in ("read_file", "edit_file", "write_file", "bash"):
        capsys.readouterr()  # clear
        ui.tool_result(name, "error: something went wrong")
        out = capsys.readouterr().out
        assert "error:" in out
        assert "something went wrong" in out


def test_drive_two_turns_shared_history(tmp_path, monkeypatch):
    """drive() over two turns sharing one messages list; second turn sees first-turn history."""
    captured = []

    def fake_llm_call(msgs, tools):
        # snapshot messages content
        captured.append([dict(m) for m in msgs])
        return {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "done", "arguments": '{"summary":"ok"}'}}]}

    monkeypatch.setattr(agent, "llm_call", fake_llm_call)
    messages = agent.new_conversation("first task", tmp_path)
    # first turn
    agent.drive(messages, tmp_path, max_steps=2)
    assert len(messages) >= 2  # system + user + assistant
    assert any("first task" in (m.get("content") or "") for m in messages)

    # second turn – append user message as REPL would
    messages.append({"role": "user", "content": "second task"})
    agent.drive(messages, tmp_path, max_steps=2)

    assert len(captured) == 2
    # second captured messages must contain both first and second task in history
    second_call_text = " ".join(m.get("content", "") for m in captured[1])
    assert "first task" in second_call_text
    assert "second task" in second_call_text
    # second call should be larger than first
    assert len(captured[1]) > len(captured[0])


def test_interactive_bare_eof_exits_without_drive(tmp_path, monkeypatch):
    """bare -i: input EOF immediately → exits without calling drive."""
    monkeypatch.setattr(agent, "llm_config", lambda: ("https://api.meta.ai/v1", "muse-spark-1.1", "k"))
    calls = []

    def fake_drive(*a, **k):
        calls.append(1)
        return "ok"

    monkeypatch.setattr(agent, "drive", fake_drive)
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    monkeypatch.setattr("sys.argv", ["agent", "--workspace", str(tmp_path), "-i"])
    agent.main()
    assert calls == []


def test_interactive_with_task_runs_once_then_eof(tmp_path, monkeypatch):
    """-i with a task: runs drive once for initial task, then EOF exits."""
    monkeypatch.setattr(agent, "llm_config", lambda: ("https://api.meta.ai/v1", "muse-spark-1.1", "k"))
    calls = []

    def fake_drive(*a, **k):
        calls.append(1)
        return "ok"

    monkeypatch.setattr(agent, "drive", fake_drive)
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    monkeypatch.setattr("sys.argv", ["agent", "--workspace", str(tmp_path), "-i", "do thing"])
    agent.main()
    assert len(calls) == 1


def test_load_memory_missing(tmp_path):
    assert agent.load_memory(tmp_path) == ""


def test_load_memory_in_system_prompt(tmp_path):
    (tmp_path / "MEMORY.md").write_text("durable fact: cats", encoding="utf-8")
    msgs = agent.new_conversation("hello", tmp_path)
    sys_msg = msgs[0]["content"]
    assert "Long-term memory" in sys_msg
    assert "durable fact: cats" in sys_msg


def test_load_memory_truncation(tmp_path):
    long_text = "x" * 5000
    (tmp_path / "MEMORY.md").write_text(long_text, encoding="utf-8")
    mem = agent.load_memory(tmp_path)
    assert len(mem) == 2000
    assert mem == long_text[:2000]
    msgs = agent.new_conversation("hi", tmp_path)
    assert long_text[:2000] in msgs[0]["content"]
    assert long_text[2000:] not in msgs[0]["content"]


def test_llm_call_surfaces_status_code(monkeypatch):
    import io
    import urllib.error as _urlerr
    import urllib.request as _urllib_req

    _llm_key(monkeypatch)

    body = b'{"error":{"message":"bad request detail too long ' + b"x" * 600 + b'"}}'

    def bad_req(req, timeout=120):
        raise _urlerr.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(body))

    monkeypatch.setattr(_urllib_req, "urlopen", bad_req)
    _stub_sleep(monkeypatch)
    try:
        agent.llm_call([{"role": "user", "content": "hi"}], [])
    except agent.ApiError as e:
        assert e.status_code == 400
        assert "Bad Request" in e.reason or "Bad Request" in str(e)
        assert len(e.body_snippet) <= 500
        assert "bad request detail" in e.body_snippet.lower() or "bad request" in e.body_snippet.lower()
    else:
        raise AssertionError("expected ApiError")


def test_drive_returns_stopped_api_on_api_error(tmp_path, monkeypatch):
    def raise_api(*a, **k):
        raise agent.ApiError(401, "Unauthorized", "invalid key snippet")

    monkeypatch.setattr(agent, "llm_call", raise_api)
    messages = agent.new_conversation("do thing", tmp_path)
    out = agent.drive(messages, tmp_path, max_steps=3)
    assert out.startswith("stopped: api")
    assert "401" in out or "Unauthorized" in out
