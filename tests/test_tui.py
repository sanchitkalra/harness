import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from rich.text import Text
from textual.widgets import Collapsible, Input, Select, Static

import model_registry as mr
import tui
from tui import TuiRenderer, _arg_hint, format_file_diff, format_status, summarize_batch


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Never let a real ~/.config/harness/models.json leak into these tests."""
    monkeypatch.setenv("HARNESS_CONFIG_DIR", str(tmp_path))


def test_format_status():
    s = format_status("muse-spark-1.1", 5, "abc123")
    assert "muse-spark-1.1" in s and "steps 5" in s and "abc123" in s

    s2 = format_status("", None, None)
    assert "unknown" in s2 and "steps 0" in s2


def test_arg_hint():
    assert _arg_hint("read_file", '{"path": "a.txt"}') == "a.txt"
    assert _arg_hint("web_search", '{"query": "foo"}') == "foo"
    assert _arg_hint("bash", '{"command": "ls"}') == "ls"
    assert _arg_hint("read_file", "not json") == ""


def test_summarize_batch():
    text, err = summarize_batch([
        ("read_file", "a.py", True),
        ("read_file", "b.py", True),
        ("bash", "pytest -q", True),
        ("bash", "pytest -x", False),
    ])
    assert text == "read 2 files: a.py, b.py · ran 2 commands: pytest -q, pytest -x · 1 error(s)"
    assert err is True
    text2, err2 = summarize_batch([("edit_file", "f.py", True)])
    assert text2 == "edited f.py"
    assert err2 is False
    text3, _ = summarize_batch([("bash", "echo hi", True)])
    assert text3 == "ran: echo hi"


def test_format_file_diff_unified():
    diff = [
        "--- a/tui.py",
        "+++ b/tui.py",
        "@@ -881,8 +881,8 @@",
        " import curses",
        "-self._setup_windows()",
        "+self._setup_windows(needed)",
        " self._draw_transcript()",
    ]
    fname, stat, styled = format_file_diff(diff, "tui.py")
    assert fname == "tui.py"
    assert stat == "Added 1 line, removed 1 line"
    assert [k for k, _ in styled] == ["dim", "del", "add", "dim"]


def test_format_file_diff_new_file():
    fname, stat, styled = format_file_diff(["+hello", "+world"], "n.txt")
    assert fname == "n.txt"
    assert stat == "Added 2 lines"
    assert all(k == "add" for k, _ in styled)
    assert styled[0][1].startswith("   1 +")


def _static_texts(app: TuiRenderer) -> list[str]:
    out = []
    for w in app.query(Static):
        r = w.content
        out.append(r.plain if isinstance(r, Text) else str(r))
    for w in app.query(Collapsible):
        out.append(str(w.title))
    return out


def test_renderer_end_to_end_smoke():
    """Drive TuiRenderer like the agent loop does (from a worker thread) and
    check the transcript/status/collapsing land in the mounted widgets."""

    async def body():
        app = TuiRenderer(model_name="test-model", session_id="sess1")
        async with app.run_test() as pilot:
            done = threading.Event()

            def work():
                app.step(1, 5, "thinking")
                app.begin_tools(1)
                app.tool_call(1, "read_file", '{"path": "a.txt"}')
                app.tool_result("read_file", "lines 1-1 of 1\nhi")
                app.tool_call(1, "edit_file", '{"path": "f.py"}')
                app.tool_result(
                    "edit_file",
                    "ok: edited f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+b",
                )
                app.tool_call(1, "bash", '{"command": "pytest -q"}')
                app.tool_result("bash", "exit=0\nall good")
                app.end_tools()
                app.final("done ok")
                app.update_status(step_count=9, session_id="sess2")
                done.set()

            threading.Thread(target=work, daemon=True).start()
            while not done.is_set():
                await pilot.pause()
                await asyncio.sleep(0.01)
            await pilot.pause()

            texts = _static_texts(app)
            joined = "\n".join(texts)
            assert "thinking" in joined
            assert "Update(f.py)" in joined
            assert "Added 1 line, removed 1 line" in joined
            assert "+ b" in joined
            assert "done ok" in joined
            # reads collapse behind a summary; edits never do
            assert any("read a.txt" in t for t in texts)
            # the collapsed summary shows the actual command, not just a count
            assert any("ran: pytest -q" in t for t in texts)
            # and expanding it still shows the original tool-call line (the bug: it didn't)
            assert any("tool: bash" in t and "pytest -q" in t for t in texts)
            assert app.model_name == "test-model"
            assert app.step_count == 9
            assert app.session_id == "sess2"

            # read_line() driven from a worker thread: submitting text unblocks it
            result: dict = {}

            def get_line():
                result["line"] = app.read_line("> ")

            t = threading.Thread(target=get_line, daemon=True)
            t.start()
            await pilot.pause()
            await pilot.click("#input")
            await pilot.press(*"hello", "enter")
            t.join(timeout=2)
            assert result["line"] == "hello"
            assert any("hello" in t for t in _static_texts(app))

    asyncio.run(body())


def test_slash_name_command_renames_session_without_reaching_the_agent():
    """/name is intercepted by read_line() and never returned as a task."""

    async def body():
        app = TuiRenderer(model_name="m", session_id="sess1")
        async with app.run_test() as pilot:
            result: dict = {}

            def get_line():
                result["line"] = app.read_line("> ")

            t = threading.Thread(target=get_line, daemon=True)
            t.start()
            await pilot.pause()
            await pilot.click("#input")
            await pilot.press(*"/name my tests", "enter")
            # slash command consumed: read_line loops, doesn't return yet
            await pilot.pause()
            await asyncio.sleep(0.05)
            assert "line" not in result
            assert app.session_name == "my tests"
            assert "session: my tests" in app._status.content.plain

            await pilot.click("#input")
            await pilot.press(*"real task", "enter")
            t.join(timeout=2)
            assert result["line"] == "real task"

    asyncio.run(body())


def test_unknown_slash_command_shows_error():
    async def body():
        app = TuiRenderer(model_name="m", session_id="sess1")
        async with app.run_test() as pilot:
            app._handle_slash_command("/bogus")
            await pilot.pause()
            assert any("unknown command: /bogus" in t for t in _static_texts(app))

    asyncio.run(body())


def test_slash_help_lists_all_commands():
    async def body():
        app = TuiRenderer(model_name="m", session_id="sess1")
        async with app.run_test() as pilot:
            app._handle_slash_command("/help")
            await pilot.pause()
            joined = "\n".join(_static_texts(app))
            for name in tui.SLASH_COMMANDS:
                assert f"/{name}" in joined

    asyncio.run(body())


def test_slash_clear_wipes_transcript_and_flags_a_context_reset():
    async def body():
        app = TuiRenderer(model_name="m", session_id="sess1")
        async with app.run_test() as pilot:
            app._mount_line("some old output", "dim")
            await pilot.pause()
            assert not app.take_clear_request()  # nothing requested yet

            app._handle_slash_command("/clear")
            await pilot.pause()
            texts = _static_texts(app)
            assert not any("some old output" in t for t in texts)
            assert any("cleared" in t for t in texts)
            # agent.py's session loop consumes this once to reset `messages`
            assert app.take_clear_request() is True
            assert app.take_clear_request() is False  # one-shot

    asyncio.run(body())


def test_slash_quit_exits_and_unblocks_read_line():
    async def body():
        app = TuiRenderer(model_name="m", session_id="sess1")
        async with app.run_test() as pilot:
            result: dict = {}

            def get_line():
                result["line"] = app.read_line("> ")

            t = threading.Thread(target=get_line, daemon=True)
            t.start()
            await pilot.pause()
            await pilot.click("#input")
            await pilot.press(*"/quit", "enter")
            t.join(timeout=2)
            assert result["line"] is None

    asyncio.run(body())


def test_hints_show_and_filter_while_typing_a_slash_command():
    async def body():
        app = TuiRenderer(model_name="m", session_id="sess1")
        async with app.run_test() as pilot:
            await pilot.click("#input")
            await pilot.press("/")
            await pilot.pause()
            assert "visible" in app._hints.classes
            for name in tui.SLASH_COMMANDS:
                assert f"/{name}" in app._hints.content.plain

            await pilot.press(*"na")  # narrows to /name (not /rename, /help, ...)
            await pilot.pause()
            hint_text = app._hints.content.plain
            assert "/name" in hint_text
            assert "/help" not in hint_text and "/rename" not in hint_text

            await pilot.press(" ")  # space: now typing the argument, hide hints
            await pilot.pause()
            assert "visible" not in app._hints.classes

    asyncio.run(body())


def test_slash_model_add_new_profile_and_it_becomes_active():
    async def body():
        app = TuiRenderer(model_name="unknown", session_id="sess1")
        async with app.run_test(size=(90, 30)) as pilot:
            app._handle_slash_command("/model")
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, tui.ModelPickerScreen)

            # no saved profiles yet: only "+ Add new profile..." is offered
            select = app.screen.query_one("#picker-select", Select)
            select.value = "__add__"
            await pilot.pause()
            assert "visible" in app.screen.query_one("#add-form").classes

            app.screen.query_one("#f-name", Input).value = "work"
            app.screen.query_one("#f-provider", Select).value = "anthropic"
            app.screen.query_one("#f-key", Input).value = "sk-ant-testkey"
            app.screen.query_one("#f-model", Input).value = "claude-sonnet-5"
            await pilot.click("#f-save")
            await pilot.pause()

            assert not isinstance(app.screen, tui.ModelPickerScreen)  # modal closed
            reg = mr.load_registry()
            assert reg["active"] == "work"
            assert reg["profiles"]["work"]["provider"] == "anthropic"
            texts = _static_texts(app)
            assert any("model set to 'work'" in t for t in texts)
            assert "session: sess1" in app._status.content.plain
            assert app.model_name == "claude-sonnet-5"

    asyncio.run(body())


def test_slash_model_save_rejects_missing_fields():
    async def body():
        app = TuiRenderer(model_name="unknown", session_id="sess1")
        async with app.run_test(size=(90, 30)) as pilot:
            app._handle_slash_command("/model")
            await pilot.pause()
            await pilot.pause()
            app.screen.query_one("#picker-select", Select).value = "__add__"
            await pilot.pause()
            await pilot.click("#f-save")  # nothing filled in
            await pilot.pause()
            assert isinstance(app.screen, tui.ModelPickerScreen)  # still open
            screen_texts = [
                w.content.plain if isinstance(w.content, Text) else str(w.content)
                for w in app.screen.query(Static)
            ]
            assert any("required" in t for t in screen_texts)
            assert mr.load_registry()["profiles"] == {}

    asyncio.run(body())


def test_slash_model_picks_existing_profile():
    async def body():
        mr.add_profile("work", "anthropic", "sk-ant-a", "claude-sonnet-5")
        mr.add_profile("cheap", "openai", "sk-oai", "gpt-4o-mini", make_active=False)
        app = TuiRenderer(model_name="unknown", session_id="sess1")
        async with app.run_test(size=(90, 30)) as pilot:
            app._handle_slash_command("/model")
            await pilot.pause()
            await pilot.pause()
            select = app.screen.query_one("#picker-select", Select)
            select.value = "cheap"
            await pilot.pause()

            assert mr.load_registry()["active"] == "cheap"
            assert any("model set to 'cheap'" in t for t in _static_texts(app))
            assert app.model_name == "gpt-4o-mini"

    asyncio.run(body())


def test_slash_model_cancel_leaves_registry_untouched():
    async def body():
        mr.add_profile("work", "anthropic", "sk-ant-a", "claude-sonnet-5")
        app = TuiRenderer(model_name="unknown", session_id="sess1")
        async with app.run_test(size=(90, 30)) as pilot:
            app._handle_slash_command("/model")
            await pilot.pause()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, tui.ModelPickerScreen)
            assert mr.load_registry()["active"] == "work"  # unchanged

    asyncio.run(body())
