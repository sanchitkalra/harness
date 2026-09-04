import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tui
from tui import (
    TranscriptBuffer,
    InputBuffer,
    format_status,
    visible_slice,
    wrap_line,
    wrap_input,
    summarize_batch,
    clamp_scroll_offset,
    handle_input_event,
    TuiRenderer,
)


def test_transcript_buffer_append_and_cap():
    buf = TranscriptBuffer(max_lines=5)
    buf.append("a")
    buf.append("b")
    assert buf.lines == ["a", "b"]
    assert len(buf) == 2

    buf.append("line1\nline2\nline3")
    assert buf.lines == ["a", "b", "line1", "line2", "line3"]
    # exceed cap
    buf.append("c4")
    assert len(buf) == 5
    # should keep newest 5: b, line1, line2, line3, c4 after capping from 6 -> 5 removes oldest a
    assert buf.lines == ["b", "line1", "line2", "line3", "c4"]

    # large overflow
    buf2 = TranscriptBuffer(max_lines=3)
    for i in range(10):
        buf2.append(f"{i}")
    assert buf2.lines == ["7", "8", "9"]
    assert len(buf2) == 3

def test_transcript_buffer_multiline_capping_edge():
    buf = TranscriptBuffer(max_lines=4)
    buf.append("x\ny")  # 2 lines
    buf.append("a\nb\nc\nd")  # adds 4 lines, total would be 6, cap to 4 => keep last 4
    assert buf.lines == ["a", "b", "c", "d"]
    # empty string handling
    buf.clear()
    buf.append("")
    assert buf.lines == [""]
    assert len(buf) == 1


def test_visible_slice_basic():
    lines = [str(i) for i in range(10)]
    # height 3, offset 0 => last 3
    assert visible_slice(lines, 3, 0) == ["7", "8", "9"]
    # offset 2 => scroll up 2
    assert visible_slice(lines, 3, 2) == ["5", "6", "7"]
    # large offset clamps to max
    assert visible_slice(lines, 3, 100) == ["0", "1", "2"]
    # negative offset clamps to 0
    assert visible_slice(lines, 3, -5) == ["7", "8", "9"]
    # height 0
    assert visible_slice(lines, 0, 0) == []


def test_get_visible_method():
    buf = TranscriptBuffer(max_lines=100)
    for i in range(10):
        buf.append(f"{i}")
    assert buf.get_visible(3, 0) == ["7", "8", "9"]
    assert buf.get_visible(3, 2) == ["5", "6", "7"]
    assert buf.get_visible(3, 999) == ["0", "1", "2"]


def test_clamp_scroll_offset():
    assert clamp_scroll_offset(10, 3, 0) == 0
    assert clamp_scroll_offset(10, 3, 2) == 2
    assert clamp_scroll_offset(10, 3, 100) == 7  # max is 10-3
    assert clamp_scroll_offset(10, 3, -1) == 0
    assert clamp_scroll_offset(2, 5, 10) == 0  # total < height -> max 0
    assert clamp_scroll_offset(0, 0, 5) == 0


def test_format_status():
    s = format_status("muse-spark-1.1", 5, "abc123")
    assert "muse-spark-1.1" in s
    assert "steps 5" in s
    assert "abc123" in s

    s2 = format_status("gpt-4o", 0, "")
    assert "gpt-4o" in s2
    assert "steps 0" in s2
    assert "abc123" not in s2

    s3 = format_status("", None, None)
    assert "unknown" in s3

    s4 = format_status("model-x", None, "sid")
    assert "steps 0" in s4
    assert "sid" in s4


def test_input_buffer_basic():
    b = InputBuffer()
    assert b.text == ""
    assert b.cursor == 0
    b.insert("a")
    assert b.text == "a"
    assert b.cursor == 1
    b.insert("bc")
    assert b.text == "abc"
    assert b.cursor == 3
    b.move_left()
    b.move_left()
    assert b.cursor == 1
    b.insert("X")
    assert b.text == "aXbc"
    b.backspace()
    assert b.text == "abc"
    assert b.cursor == 1
    b.move_end()
    assert b.cursor == 3
    b.move_home()
    assert b.cursor == 0
    b.delete()  # delete 'a'
    assert b.text == "bc"
    b.clear()
    assert b.text == ""
    assert b.cursor == 0


def test_handle_input_event():
    buf = InputBuffer()
    submit, exit_sig = handle_input_event(buf, "char", "h")
    assert not submit and not exit_sig
    assert buf.text == "h"

    handle_input_event(buf, "char", "i")
    assert buf.text == "hi"

    submit, exit_sig = handle_input_event(buf, "left")
    assert buf.cursor == 1

    submit, exit_sig = handle_input_event(buf, "backspace")
    assert buf.text == "i"

    buf.set_text("hello")
    handle_input_event(buf, "home")
    assert buf.cursor == 0
    handle_input_event(buf, "end")
    assert buf.cursor == 5
    handle_input_event(buf, "ctrl_u")
    assert buf.text == ""

    buf.set_text("hello world")
    buf.cursor = 5
    handle_input_event(buf, "ctrl_k")
    assert buf.text == "hello"

    buf.set_text("test")
    submit, exit_sig = handle_input_event(buf, "enter")
    assert submit is True and exit_sig is False

    submit, exit_sig = handle_input_event(buf, "ctrl_d")
    assert submit is False and exit_sig is True

    submit, exit_sig = handle_input_event(buf, "ctrl_c")
    assert exit_sig is True

    # unknown key ignored
    buf2 = InputBuffer("abc")
    submit, exit_sig = handle_input_event(buf2, "unknown_key")
    assert buf2.text == "abc"
    assert not submit and not exit_sig


def test_tuirenderer_protocol_and_state():
    # Ensure TuiRenderer implements protocol without curses
    r = TuiRenderer(model_name="test-model", session_id="sess123", max_lines=10)
    assert r.model_name == "test-model"
    assert r.session_id == "sess123"

    # step
    r.step(1, 10, "thinking")
    assert any("thinking" in line for line in r.transcript.lines)
    assert r.step_count == 1

    r.tool_call(1, "read_file", '{"path": "a.txt"}')
    assert any("read_file" in line for line in r.transcript.lines)

    r.tool_result("read_file", "hello world")
    assert any("hello world" in line for line in r.transcript.lines)

    r.begin_tools(2)
    assert r.step_count == 2

    r.end_tools()  # should not raise

    r.final("done ok")
    assert any("done ok" in line for line in r.transcript.lines)

    # transcript cap enforcement via many lines
    r2 = TuiRenderer(max_lines=3)
    r2.step(1, 1, "a")
    r2.step(2, 1, "b")
    r2.step(3, 1, "c")
    r2.step(4, 1, "d")
    assert len(r2.transcript) == 3
    assert r2.transcript.lines[-1].endswith("d")

    # update_status pure
    r.update_status(model_name="new-model", step_count=7, session_id="newid")
    assert r.model_name == "new-model"
    assert r.step_count == 7
    assert r.session_id == "newid"


def test_wrap_line():
    assert wrap_line("hello world foo", 8) == ["hello ", "world ", "foo"]
    assert wrap_line("", 10) == [""]
    assert wrap_line("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]
    # no content lost across chunks
    assert "".join(wrap_line("ab cd ef gh", 5)) == "ab cd ef gh"


def test_wrap_input_single_row():
    rows, crow, ccol = wrap_input("> ", "hi", 2, 20, 3)
    assert rows == ["> hi"]
    assert (crow, ccol) == (0, 4)


def test_wrap_input_long_prompt_wraps_and_tracks_cursor():
    text = "word " * 20  # 100 chars
    rows, crow, ccol = wrap_input("> ", text, len(text), 20, 3)
    assert len(rows) == 3  # scrolled to cursor row
    assert crow == 2
    assert rows[-1].strip() == "word"  # tail with cursor visible
    assert 0 <= ccol <= len(rows[crow])


def test_wrap_input_midline_cursor():
    rows, crow, ccol = wrap_input("> ", "abcdefghij", 3, 20, 3)
    assert rows == ["> abcdefghij"]
    assert (crow, ccol) == (0, 2 + 3)


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


def test_transcript_kinds_and_collapse():
    from tui import K_OK
    buf = TranscriptBuffer(max_lines=10)
    buf.append("Step 1/5 hi", "step")
    buf.append("tool line", "tool")
    assert buf.items() == [("step", "Step 1/5 hi"), ("tool", "tool line")]
    assert buf.lines == ["Step 1/5 hi", "tool line"]  # plain view unchanged
    buf.collapse_from(1, [(K_OK, "✔ ↳ read a.py")])
    assert buf.lines == ["Step 1/5 hi", "✔ ↳ read a.py"]
    assert buf.items()[1][0] == K_OK
    buf.clear()
    assert buf.lines == [] and buf.items() == []


def test_renderer_collapses_batch_to_summary():
    r = TuiRenderer(model_name="m", session_id="s")
    r.step(1, 5, "do things")
    r.begin_tools(1)
    r.tool_call(1, "read_file", '{"path": "a.txt"}')
    r.tool_result("read_file", "lines 1-2 of 2\nhi")
    r.tool_call(1, "bash", '{"command": "echo hi"}')
    r.tool_result("bash", "exit=0")
    r.end_tools()
    lines = r.transcript.lines
    assert any(l.startswith("Step 1/5") for l in lines)
    assert any("↳" in l and "read a.txt" in l and "ran 1 command" in l for l in lines)
    assert not any("exit=0" in l for l in lines)  # live lines collapsed away
    assert any("✔" in l for l in lines)


def test_renderer_error_summary():
    r = TuiRenderer(model_name="m", session_id="s")
    r.begin_tools(1)
    r.tool_call(1, "bash", '{"command": "rm -rf / x"}')
    r.tool_result("bash", "error: blocked: nope")
    r.end_tools()
    summaries = [l for l in r.transcript.lines if "↳" in l]
    assert len(summaries) == 1
    assert "✖" in summaries[0] and "1 error(s)" in summaries[0]


def test_input_height_for():
    from tui import input_height_for
    assert input_height_for(1, 24) == 1
    assert input_height_for(10, 24) == 10
    # capped so transcript keeps >=3 rows + status
    assert input_height_for(100, 24) == 24 - 1 - 3
    assert input_height_for(100, 6) == 6 - 1 - 3
    assert input_height_for(5, 4) == 1  # tiny screen: floor of 1


def test_toggle_expand_collapse():
    r = TuiRenderer(model_name="m", session_id="s")
    r.step(1, 5, "work")
    r.begin_tools(1)
    r.tool_call(1, "read_file", '{"path": "a.txt"}')
    r.tool_result("read_file", "lines 1-2 of 2\nhi")
    r.end_tools()
    assert len(r._collapsed) == 1
    idx = next(iter(r._collapsed))
    assert r.transcript.lines[idx].endswith("[+]")
    # expand: details back inline, marker flips
    assert r.toggle_at_index(idx) is True
    lines = r.transcript.lines
    assert any("exit=0" in l or "hi" in l for l in lines) or any("lines 1-2" in l for l in lines)
    assert r.transcript.lines[idx].endswith("[–]")
    # collapse again
    assert r.toggle_at_index(idx) is True
    assert r.transcript.lines[idx].endswith("[+]")
    assert not any("lines 1-2" in l for l in r.transcript.lines)


def test_toggle_unknown_and_stale():
    r = TuiRenderer(model_name="m", session_id="s")
    assert r.toggle_at_index(99) is False
    assert r.toggle_at_row(0) is False  # nothing drawn
    r.begin_tools(1)
    r.tool_call(1, "bash", '{"command": "echo hi"}')
    r.tool_result("bash", "exit=0")
    r.end_tools()
    idx = next(iter(r._collapsed))
    r.transcript.clear()  # cap-like shift: summary gone
    assert r.toggle_at_index(idx) is False  # stale guard, no crash


def test_toggle_row_mapping():
    r = TuiRenderer(model_name="m", session_id="s")
    r.transcript.append("plain line")
    r.begin_tools(1)
    r.tool_call(1, "bash", '{"command": "echo hi"}')
    r.tool_result("bash", "exit=0")
    r.end_tools()
    idx = next(iter(r._collapsed))
    assert idx == 1
    r._last_drawn = [0, 1]  # fabricated draw map: row 0 plain, row 1 summary
    assert r.toggle_at_row(1) is True
    assert r.toggle_at_row(0) is False  # plain line, no record
    assert r.toggle_at_row(99) is False


def test_format_file_diff_unified():
    from tui import format_file_diff, K_ADDROW, K_DELROW, K_DIM
    diff = [
        "--- a/tui.py",
        "+++ b/tui.py",
        "@@ -881,8 +881,8 @@",
        " import curses",
        "-self._setup_windows()",
        "+needed = len(wrap_line(x))",
        "+self._setup_windows(needed)",
        " self._draw_transcript()",
    ]
    fname, stat, styled = format_file_diff(diff, "tui.py")
    assert fname == "tui.py"
    assert stat == "Added 2 lines, removed 1 line"
    kinds = [k for k, _ in styled]
    assert kinds == [K_DIM, K_DELROW, K_ADDROW, K_ADDROW, K_DIM]
    texts = [t for _, t in styled]
    assert texts[0] == " 881   import curses"
    assert texts[1] == " 882 - self._setup_windows()"
    assert texts[2] == " 882 + needed = len(wrap_line(x))"
    assert texts[3] == " 883 + self._setup_windows(needed)"


def test_format_file_diff_singular_and_new_file():
    from tui import format_file_diff, K_ADDROW
    fname, stat, styled = format_file_diff(["+hello", "+world"], "n.txt")
    assert fname == "n.txt"
    assert stat == "Added 2 lines"
    assert all(k == K_ADDROW for k, _ in styled)
    assert styled[0][1].startswith("   1 +")
    fname2, stat2, _ = format_file_diff(["+only"], "one.txt")
    assert stat2 == "Added 1 line"


def test_renderer_edit_block():
    from tui import K_HEAD, K_DIM
    r = TuiRenderer(model_name="m", session_id="s")
    r.begin_tools(1)
    r.tool_call(1, "edit_file", '{"path": "f.py", "find": "a", "replace": "b"}')
    r.tool_result("edit_file", "ok: edited f.py\n--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n line1\n-line2\n+LINE2\n line3")
    pre = r.transcript.lines
    assert "Update(f.py)" in pre
    assert any("Added 1 line, removed 1 line" in l for l in pre)
    assert any("LINE2" in l for l in pre)
    kinds = dict((t, k) for k, t in r.transcript.items())
    assert kinds["Update(f.py)"] == K_HEAD
    r.end_tools()
    lines = r.transcript.lines
    # collapse replaces the block with one summary
    assert sum(1 for l in lines if "↳" in l) == 1
    assert not any(l.startswith("  -> edit_file") for l in lines)
    # click restores the styled block
    assert r.toggle_at_index(next(iter(r._collapsed))) is True
    assert "Update(f.py)" in r.transcript.lines


def test_boxed_input_width_and_kinds():
    from tui import boxed_input, K_DIM, K_INPUT
    box = boxed_input("hello world foo bar", 20)
    assert all(len(t) <= 20 for _, t in box)
    assert box[0] == (K_DIM, "┌" + "─" * 18 + "┐")
    assert box[-1] == (K_DIM, "└" + "─" * 18 + "┘")
    assert all(k == K_INPUT for k, _ in box[1:-1])
    # content preserved across wrapped rows
    inner = "".join(t[2:-2].rstrip() for _, t in box[1:-1]).replace("  ", " ")
    assert "hello" in inner and "bar" in inner


def test_boxed_input_empty():
    from tui import boxed_input
    box = boxed_input("", 20)
    assert len(box) == 3  # top, one empty row, bottom
    assert all(len(t) <= 20 for _, t in box)
