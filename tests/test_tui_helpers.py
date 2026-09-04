import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tui
from tui import (
    TranscriptBuffer,
    InputBuffer,
    format_status,
    visible_slice,
    visible_input,
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


def test_visible_input_fits():
    display, cx = visible_input("> ", "hi", 2, 20)
    assert display == "> hi"
    assert cx == 4


def test_visible_input_scrolls_to_cursor_at_end():
    text = "x" * 50
    display, cx = visible_input("> ", text, 50, 20)
    assert len(display) <= 20
    assert display.endswith("x")
    assert cx == len(display)  # cursor visible at end


def test_visible_input_cursor_midline_stays_visible():
    text = "a" * 50
    display, cx = visible_input("> ", text, 25, 20)
    assert display == "> " + "a" * 17
    assert cx == 18  # cursor column inside the shown window


def test_visible_input_short_text_no_scroll():
    display, cx = visible_input("> ", "abc", 1, 20)
    assert display == "> abc"
    assert cx == len("> ") + 1
