import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.text import Text
from textual.widgets import Collapsible, Static

from tui import TuiRenderer, _arg_hint, format_file_diff, format_status, summarize_batch


def test_format_status():
    s = format_status("muse-spark-1.1", 5, "abc123")
    assert "muse-spark-1.1" in s and "steps 5" in s and "abc123" in s

    s2 = format_status("", None, None)
    assert "unknown" in s2 and "steps 0" in s2


def test_arg_hint():
    assert _arg_hint("read_file", '{"path": "a.txt"}') == "a.txt"
    assert _arg_hint("web_search", '{"query": "foo"}') == "foo"
    assert _arg_hint("bash", '{"command": "ls"}') == ""
    assert _arg_hint("read_file", "not json") == ""


def test_summarize_batch():
    text, err = summarize_batch([
        ("read_file", "a.py", True),
        ("read_file", "b.py", True),
        ("bash", "", True),
        ("bash", "", False),
    ])
    assert text == "read 2 files: a.py, b.py · ran 2 commands · 1 error(s)"
    assert err is True
    text2, err2 = summarize_batch([("edit_file", "f.py", True)])
    assert text2 == "edited f.py"
    assert err2 is False


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
