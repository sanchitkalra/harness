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
from tui import TuiRenderer, _arg_hint, format_file_diff, format_footer, format_status, run_title, summarize_batch


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Never let a real ~/.config/harness/models.json leak into these tests."""
    monkeypatch.setenv("HARNESS_CONFIG_DIR", str(tmp_path))


def test_format_status():
    assert format_status(5, 30) == "step 5/30"
    assert format_status(0) == "step 0"


def test_format_footer():
    session, model = format_footer("my session", "sess1", "work", "claude-sonnet-5")
    assert session == "my session"
    assert model == "work/claude-sonnet-5"

    session2, model2 = format_footer("", "sess1", "", "")
    assert session2 == "sess1"
    assert model2 == "unknown"


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


def test_run_title_paginated_reads_of_same_file():
    title = run_title("read_file", ["pr7590.diff"] * 11, False)
    assert title == "✔ read pr7590.diff (11 reads)"


def test_run_title_multiple_distinct_files():
    title = run_title("read_file", ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"], False)
    assert title == "✔ read 6 files: a.py, b.py, c.py, d.py +2 more"


def test_run_title_single_read():
    assert run_title("read_file", ["a.py"], False) == "✔ read a.py"


def test_run_title_bash_commands():
    assert run_title("bash", ["ls", "pytest -q"], False) == "✔ ran 2 commands"
    assert run_title("bash", ["ls"], True) == "✖ ran 1 command"


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
            # no separate "Step i/n" line in the transcript anymore
            assert "Step 1/5" not in joined
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


def test_multiple_reads_in_one_step_collapse_into_one_running_line():
    async def body():
        app = TuiRenderer(model_name="m", session_id="sess1")
        async with app.run_test() as pilot:
            done = threading.Event()

            def work():
                app.begin_tools(1)
                app.tool_call(1, "read_file", '{"path": "a.py"}')
                app.tool_result("read_file", "lines 1-1 of 1\nx")
                app.tool_call(1, "read_file", '{"path": "b.py"}')
                app.tool_result("read_file", "lines 1-1 of 1\ny")
                app.tool_call(1, "read_file", '{"path": "c.py"}')
                app.tool_result("read_file", "lines 1-1 of 1\nz")
                done.set()

            threading.Thread(target=work, daemon=True).start()
            while not done.is_set():
                await pilot.pause()
                await asyncio.sleep(0.01)
            await pilot.pause()

            texts = _static_texts(app)
            assert any(t == "  tool: read a.py, b.py, c.py" for t in texts)
            # no per-read result lines, and no separate per-file "tool: read" lines
            assert sum(1 for t in texts if t.startswith("  tool: read")) == 1

    asyncio.run(body())


def test_paginated_reads_across_separate_steps_collapse_into_one_collapsible():
    """The reported bug: a paginated read (one read_file call per agent
    step, e.g. re-reading the same file with offset=) used to leave one
    "read x" collapsible per step instead of collapsing into one."""

    async def body():
        app = TuiRenderer(model_name="m", session_id="sess1")
        async with app.run_test() as pilot:
            done = threading.Event()

            def work():
                for _ in range(11):
                    app.begin_tools(1)
                    app.tool_call(1, "read_file", '{"path": "pr7590.diff"}')
                    app.tool_result("read_file", "lines 1-500 of 5000\n...")
                    app.end_tools()
                done.set()

            threading.Thread(target=work, daemon=True).start()
            while not done.is_set():
                await pilot.pause()
                await asyncio.sleep(0.01)
            await pilot.pause()

            collapsibles = list(app.query(Collapsible))
            assert len(collapsibles) == 1
            assert str(collapsibles[0].title) == "✔ read pr7590.diff (11 reads)"
            # expanding it still shows every individual call, not just the last
            detail_texts = [
                w.content.plain if isinstance(w.content, Text) else str(w.content)
                for w in collapsibles[0].query(Static)
            ]
            assert sum(1 for t in detail_texts if "tool: read" in t) == 11

    asyncio.run(body())


def test_repeated_bash_calls_across_steps_collapse_and_expand():
    async def body():
        app = TuiRenderer(model_name="m", session_id="sess1")
        async with app.run_test() as pilot:
            done = threading.Event()

            def work():
                for cmd in ("pytest -q", "pytest -x", "echo done"):
                    app.begin_tools(1)
                    app.tool_call(1, "bash", f'{{"command": "{cmd}"}}')
                    app.tool_result("bash", "exit=0\nok")
                    app.end_tools()
                done.set()

            threading.Thread(target=work, daemon=True).start()
            while not done.is_set():
                await pilot.pause()
                await asyncio.sleep(0.01)
            await pilot.pause()

            collapsibles = list(app.query(Collapsible))
            assert len(collapsibles) == 1
            assert str(collapsibles[0].title) == "✔ ran 3 commands"
            detail_texts = [
                w.content.plain if isinstance(w.content, Text) else str(w.content)
                for w in collapsibles[0].query(Static)
            ]
            assert any("pytest -q" in t for t in detail_texts)
            assert any("pytest -x" in t for t in detail_texts)
            assert any("echo done" in t for t in detail_texts)

    asyncio.run(body())


def test_bash_error_marks_the_whole_run_as_errored():
    async def body():
        app = TuiRenderer(model_name="m", session_id="sess1")
        async with app.run_test() as pilot:
            done = threading.Event()

            def work():
                app.begin_tools(1)
                app.tool_call(1, "bash", '{"command": "ls"}')
                app.tool_result("bash", "exit=0\nok")
                app.end_tools()
                app.begin_tools(2)
                app.tool_call(2, "bash", '{"command": "pytest -q"}')
                app.tool_result("bash", "error: exit=1\nboom")
                app.end_tools()
                done.set()

            threading.Thread(target=work, daemon=True).start()
            while not done.is_set():
                await pilot.pause()
                await asyncio.sleep(0.01)
            await pilot.pause()

            collapsibles = list(app.query(Collapsible))
            assert len(collapsibles) == 1
            assert str(collapsibles[0].title) == "✖ ran 2 commands"

    asyncio.run(body())


def test_a_different_tool_kind_breaks_the_run():
    async def body():
        app = TuiRenderer(model_name="m", session_id="sess1")
        async with app.run_test() as pilot:
            done = threading.Event()

            def work():
                app.begin_tools(1)
                app.tool_call(1, "read_file", '{"path": "a.py"}')
                app.tool_result("read_file", "lines 1-1 of 1\nx")
                app.end_tools()
                app.begin_tools(2)
                app.tool_call(2, "bash", '{"command": "ls"}')
                app.tool_result("bash", "exit=0\nok")
                app.end_tools()
                app.begin_tools(3)
                app.tool_call(3, "read_file", '{"path": "b.py"}')
                app.tool_result("read_file", "lines 1-1 of 1\ny")
                app.end_tools()
                done.set()

            threading.Thread(target=work, daemon=True).start()
            while not done.is_set():
                await pilot.pause()
                await asyncio.sleep(0.01)
            await pilot.pause()

            collapsibles = list(app.query(Collapsible))
            # read run, then bash run, then a fresh read run — not merged across the bash
            assert [str(c.title) for c in collapsibles] == ["✔ read a.py", "✔ ran 1 command", "✔ read b.py"]

    asyncio.run(body())


def test_footer_shows_session_name_and_group_model():
    async def body():
        app = TuiRenderer(model_name="claude-sonnet-5", session_id="sess1")
        app.group_name = "work"
        async with app.run_test() as pilot:
            app._refresh_status()
            await pilot.pause()
            assert app._footer_session.content.plain == "sess1"
            assert app._footer_model.content.plain == "work/claude-sonnet-5"

            app._handle_slash_command("/name my session")
            await pilot.pause()
            assert app._footer_session.content.plain == "my session"

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
            assert app._footer_session.content.plain == "my tests"

            await pilot.click("#input")
            await pilot.press(*"real task", "enter")
            t.join(timeout=2)
            assert result["line"] == "real task"

    asyncio.run(body())


def test_shift_enter_inserts_newline_plain_enter_submits():
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
            await pilot.press(*"line one")
            await pilot.press("shift+enter")
            await pilot.press(*"line two")
            await pilot.pause()
            assert app._input.text == "line one\nline two"
            assert "line" not in result  # not submitted yet

            await pilot.press("enter")
            t.join(timeout=2)
            assert result["line"] == "line one\nline two"
            assert app._input.text == ""

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


def test_slash_login_add_new_group_and_it_becomes_active():
    async def body():
        app = TuiRenderer(model_name="unknown", session_id="sess1")
        async with app.run_test(size=(90, 40)) as pilot:
            app._handle_slash_command("/login")
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, tui.LoginPickerScreen)

            # no saved logins yet: only "+ Add new login..." is offered
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

            assert not isinstance(app.screen, tui.LoginPickerScreen)  # modal closed
            reg = mr.load_registry()
            assert reg["active_group"] == "work"
            assert reg["groups"]["work"]["provider"] == "anthropic"
            texts = _static_texts(app)
            assert any("login set to 'work'" in t for t in texts)
            assert app.group_name == "work"
            assert app.model_name == "claude-sonnet-5"

    asyncio.run(body())


def test_slash_login_compatible_provider_requires_base_url():
    async def body():
        app = TuiRenderer(model_name="unknown", session_id="sess1")
        async with app.run_test(size=(90, 40)) as pilot:
            app._handle_slash_command("/login")
            await pilot.pause()
            await pilot.pause()
            app.screen.query_one("#picker-select", Select).value = "__add__"
            await pilot.pause()
            app.screen.query_one("#f-name", Input).value = "local"
            app.screen.query_one("#f-provider", Select).value = "compatible"
            app.screen.query_one("#f-key", Input).value = "k"
            app.screen.query_one("#f-model", Input).value = "some-model"
            await pilot.click("#f-save")
            await pilot.pause()

            assert isinstance(app.screen, tui.LoginPickerScreen)  # still open
            screen_texts = [
                w.content.plain if isinstance(w.content, Text) else str(w.content)
                for w in app.screen.query(Static)
            ]
            assert any("base url is required" in t for t in screen_texts)
            assert mr.load_registry()["groups"] == {}

    asyncio.run(body())


def test_slash_login_save_rejects_missing_fields():
    async def body():
        app = TuiRenderer(model_name="unknown", session_id="sess1")
        async with app.run_test(size=(90, 40)) as pilot:
            app._handle_slash_command("/login")
            await pilot.pause()
            await pilot.pause()
            app.screen.query_one("#picker-select", Select).value = "__add__"
            await pilot.pause()
            await pilot.click("#f-save")  # nothing filled in
            await pilot.pause()
            assert isinstance(app.screen, tui.LoginPickerScreen)  # still open
            screen_texts = [
                w.content.plain if isinstance(w.content, Text) else str(w.content)
                for w in app.screen.query(Static)
            ]
            assert any("required" in t for t in screen_texts)
            assert mr.load_registry()["groups"] == {}

    asyncio.run(body())


def test_slash_login_picks_existing_group():
    async def body():
        mr.add_group("work", "anthropic", "sk-ant-a", "claude-sonnet-5")
        mr.add_group("cheap", "openai", "sk-oai", "gpt-4o-mini", make_active=False)
        app = TuiRenderer(model_name="unknown", session_id="sess1")
        async with app.run_test(size=(90, 40)) as pilot:
            app._handle_slash_command("/login")
            await pilot.pause()
            await pilot.pause()
            select = app.screen.query_one("#picker-select", Select)
            select.value = "cheap"
            await pilot.pause()

            assert mr.load_registry()["active_group"] == "cheap"
            assert any("login set to 'cheap'" in t for t in _static_texts(app))
            assert app.group_name == "cheap"
            assert app.model_name == "gpt-4o-mini"

    asyncio.run(body())


def test_slash_login_cancel_leaves_registry_untouched():
    async def body():
        mr.add_group("work", "anthropic", "sk-ant-a", "claude-sonnet-5")
        app = TuiRenderer(model_name="unknown", session_id="sess1")
        async with app.run_test(size=(90, 40)) as pilot:
            app._handle_slash_command("/login")
            await pilot.pause()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, tui.LoginPickerScreen)
            assert mr.load_registry()["active_group"] == "work"  # unchanged

    asyncio.run(body())


def test_slash_model_requires_a_login_first():
    async def body():
        app = TuiRenderer(model_name="unknown", session_id="sess1")
        async with app.run_test(size=(90, 40)) as pilot:
            app._handle_slash_command("/model")
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, tui.ModelPickerScreen)
            screen_texts = [
                w.content.plain if isinstance(w.content, Text) else str(w.content)
                for w in app.screen.query(Static)
            ]
            assert any("run /login first" in t for t in screen_texts)

    asyncio.run(body())


def test_slash_model_add_new_model_to_active_login():
    async def body():
        mr.add_group("work", "anthropic", "sk-ant-a", "claude-sonnet-5")
        app = TuiRenderer(model_name="claude-sonnet-5", session_id="sess1")
        app.group_name = "work"
        async with app.run_test(size=(90, 40)) as pilot:
            app._handle_slash_command("/model")
            await pilot.pause()
            await pilot.pause()
            app.screen.query_one("#picker-select", Select).value = "__add__"
            await pilot.pause()
            app.screen.query_one("#f-model", Input).value = "claude-opus-5"
            await pilot.click("#f-save")
            await pilot.pause()

            assert not isinstance(app.screen, tui.ModelPickerScreen)
            group = mr.get_active_group()
            assert group["models"] == ["claude-sonnet-5", "claude-opus-5"]
            assert group["active_model"] == "claude-opus-5"
            assert app.model_name == "claude-opus-5"

    asyncio.run(body())


def test_slash_model_picks_existing_model_within_active_login():
    async def body():
        mr.add_group("work", "anthropic", "sk-ant-a", "claude-sonnet-5")
        mr.add_model("work", "claude-opus-5", make_active=False)
        app = TuiRenderer(model_name="claude-sonnet-5", session_id="sess1")
        app.group_name = "work"
        async with app.run_test(size=(90, 40)) as pilot:
            app._handle_slash_command("/model")
            await pilot.pause()
            await pilot.pause()
            app.screen.query_one("#picker-select", Select).value = "claude-opus-5"
            await pilot.pause()

            assert mr.get_active_group()["active_model"] == "claude-opus-5"
            assert app.model_name == "claude-opus-5"

    asyncio.run(body())
