"""Curses TUI for mini coding-agent — stdlib curses only, macOS/Linux.

Provides:
- pure helpers: TranscriptBuffer, InputBuffer, format_status, visible_slice helpers, handle_input_event
- TuiRenderer: implements ui.Renderer protocol with alternate-screen, scrollable transcript, single-line input, status bar

Curses calls are isolated in TuiRenderer methods; pure helpers are terminal-free.
"""
from __future__ import annotations

import json
import sys
import textwrap
from collections.abc import Iterable

# Line kinds for colored transcript rendering.
K_STEP = "step"
K_TOOL = "tool"
K_OK = "ok"
K_ERR = "err"
K_ADD = "add"
K_DEL = "del"
K_HUNK = "hunk"
K_DIM = "dim"
K_INPUT = "input"
K_RESULT = "result"
K_SUMMARY = "summary"
K_SUMMARY_ERR = "summary_err"
K_PLAIN = "plain"

# Input box height in rows (soft-wrapped, single logical line).
INPUT_ROWS = 3


def wrap_line(line: str, width: int) -> list[str]:
    """Pure word-wrap of one line to width. Never returns empty list."""
    line = (line or "").expandtabs(4)
    width = max(4, width)
    chunks = textwrap.wrap(
        line, width, break_long_words=True, break_on_hyphens=False,
        drop_whitespace=False, replace_whitespace=False,
    )
    return chunks or [""]


def wrap_input(prompt: str, text: str, cursor: int, width: int, max_rows: int) -> tuple[list[str], int, int]:
    """Pure soft-wrap for the input box. Returns (rows, cursor_row, cursor_col),
    rows scrolled so the cursor row is visible."""
    prompt = prompt or ""
    text = text or ""
    cursor = max(0, min(cursor, len(text)))
    width = max(8, width)
    max_rows = max(1, max_rows)
    full = prompt + text
    cpos = len(prompt) + cursor
    rows = wrap_line(full, width)
    # map absolute cursor pos to (row, col); chunks concatenate back to full
    pos = 0
    crow, ccol = len(rows) - 1, len(rows[-1])
    for i, ch in enumerate(rows):
        if cpos <= pos + len(ch):
            crow, ccol = i, cpos - pos
            break
        pos += len(ch)
    if len(rows) > max_rows:
        start = max(0, min(crow, len(rows) - max_rows))
        rows = rows[start:start + max_rows]
        crow -= start
    return rows, crow, ccol


def summarize_batch(entries: list[tuple[str, str, bool]]) -> tuple[str, bool]:
    """Pure per-step summary. entries = (tool_name, arg_hint, ok).
    Returns (summary_text, has_error). Mirrors the print renderer's wording."""
    reads: list[str] = []
    edits: list[str] = []
    writes: list[str] = []
    bashes = 0
    searches: list[str] = []
    others: dict[str, int] = {}
    errors = 0
    for name, hint, ok in entries:
        if not ok:
            errors += 1
        if name == "read_file":
            reads.append(hint or "?")
        elif name == "edit_file":
            edits.append(hint or "?")
        elif name == "write_file":
            writes.append(hint or "?")
        elif name == "bash":
            bashes += 1
        elif name == "web_search":
            searches.append(hint or "?")
        elif name == "done":
            continue
        else:
            others[name] = others.get(name, 0) + 1

    def files(word: str, fs: list[str]) -> str:
        if len(fs) == 1:
            return f"{word} {fs[0]}"
        uniq = list(dict.fromkeys(fs))[:4]
        more = f" +{len(fs) - len(uniq)} more" if len(fs) > 4 else ""
        return f"{word} {len(fs)} files: {', '.join(uniq)}{more}"

    parts: list[str] = []
    if reads:
        parts.append(files("read", reads))
    if edits:
        parts.append(files("edited", edits))
    if writes:
        parts.append(files("wrote", writes))
    if bashes:
        parts.append("ran 1 command" if bashes == 1 else f"ran {bashes} commands")
    if searches:
        parts.append(f"searched {searches[0]!r}"[:80] if len(searches) == 1 else f"searched {len(searches)} queries")
    for k, v in others.items():
        parts.append(k if v == 1 else f"{k} x{v}")
    if errors:
        parts.append(f"{errors} error(s)")
    return (" · ".join(parts)) or "tools", errors > 0

# ----------------------------------------------------------------------
# Pure helpers (no curses)
# ----------------------------------------------------------------------

class TranscriptBuffer:
    """Bounded transcript buffer — keeps last max_lines lines.

    Each line carries a kind (see K_* constants) for colored rendering.
    lines/get_visible keep the old plain-string behavior; items/
    get_visible_items expose (kind, text) tuples.
    """

    def __init__(self, max_lines: int = 2000):
        if max_lines <= 0:
            max_lines = 2000
        self.max_lines = max_lines
        self._lines: list[str] = []
        self._kinds: list[str] = []

    def append(self, text: str, kind: str = K_PLAIN) -> None:
        """Append text (may be multiline) as separate lines, then cap."""
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        parts = text.splitlines() or [""]
        for p in parts:
            self._lines.append(p)
            self._kinds.append(kind)
        self._cap()

    def _cap(self) -> None:
        if len(self._lines) > self.max_lines:
            cut = len(self._lines) - self.max_lines
            del self._lines[:cut]
            del self._kinds[:cut]

    def collapse_from(self, start: int, summary: list[tuple[str, str]]) -> None:
        """Delete lines from start onward, replace with summary (kind, text) items."""
        start = max(0, start)
        del self._lines[start:]
        del self._kinds[start:]
        for kind, text in summary:
            self._lines.append(text)
            self._kinds.append(kind)
        self._cap()

    def extend(self, lines: Iterable[str]) -> None:
        for ln in lines:
            self.append(ln)

    def clear(self) -> None:
        self._lines.clear()
        self._kinds.clear()

    @property
    def lines(self) -> list[str]:
        return list(self._lines)

    def items(self) -> list[tuple[str, str]]:
        """(kind, text) pairs in order."""
        return list(zip(self._kinds, self._lines))

    def __len__(self) -> int:
        return len(self._lines)

    def get_visible(self, height: int, scroll_offset: int = 0) -> list[str]:
        """Return visible slice for a window of given height.
        scroll_offset = 0 means bottom is visible (auto-scroll). Positive means scrolled up.
        """
        return visible_slice(self._lines, height, scroll_offset)

    def get_visible_items(self, height: int, scroll_offset: int = 0) -> list[tuple[str, str]]:
        """Kind-annotated version of get_visible."""
        items = self.items()
        if height <= 0 or not items:
            return []
        total = len(items)
        max_offset = max(0, total - height)
        off = min(max(0, scroll_offset), max_offset)
        start = max(0, total - height - off)
        return items[start:start + height]


def visible_slice(lines: list[str], height: int, scroll_offset: int = 0) -> list[str]:
    """Pure function version of get_visible for arbitrary list."""
    if height <= 0:
        return []
    total = len(lines)
    if total == 0:
        return []
    max_offset = max(0, total - height)
    off = scroll_offset
    if off < 0:
        off = 0
    if off > max_offset:
        off = max_offset
    start = max(0, total - height - off)
    end = min(total, start + height)
    return lines[start:end]


def clamp_scroll_offset(total_lines: int, height: int, offset: int) -> int:
    """Clamp scroll offset to valid range."""
    if height <= 0:
        return 0
    max_off = max(0, total_lines - height)
    if offset < 0:
        return 0
    if offset > max_off:
        return max_off
    return offset


def format_status(model_name: str, step_count: int | None = None, session_id: str | None = None) -> str:
    """Pure status-line formatter: 'model | steps N | session_id'."""
    model = (model_name or "unknown")
    model = str(model).strip() or "unknown"
    steps = 0
    if step_count is not None:
        try:
            steps = int(step_count)
        except Exception:
            steps = 0
    sid = (session_id or "").strip()
    if sid:
        return f"{model} | steps {steps} | {sid}"
    else:
        return f"{model} | steps {steps}"


class InputBuffer:
    """Pure single-line input buffer with cursor."""

    def __init__(self, text: str = ""):
        self.text: str = text
        self.cursor: int = len(text)

    def insert(self, ch: str) -> None:
        if not ch:
            return
        # ensure single char insertion but allow string of length >1
        # Insert at cursor
        self.text = self.text[: self.cursor] + ch + self.text[self.cursor :]
        self.cursor += len(ch)

    def backspace(self) -> None:
        if self.cursor > 0:
            self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
            self.cursor -= 1

    def delete(self) -> None:
        if self.cursor < len(self.text):
            self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]

    def move_left(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def move_right(self) -> None:
        if self.cursor < len(self.text):
            self.cursor += 1

    def move_home(self) -> None:
        self.cursor = 0

    def move_end(self) -> None:
        self.cursor = len(self.text)

    def clear(self) -> None:
        self.text = ""
        self.cursor = 0

    def set_text(self, t: str) -> None:
        self.text = t
        self.cursor = len(t)

    def __repr__(self) -> str:
        return f"InputBuffer(text={self.text!r}, cursor={self.cursor})"


# For input processing tests: maps logical keys to buffer actions.
# Returns (should_submit, should_exit)
def handle_input_event(buf: InputBuffer, key: str, char: str | None = None) -> tuple[bool, bool]:
    """
    Pure input handling.

    key: one of
      'char', 'backspace', 'delete', 'left', 'right', 'home', 'end',
      'enter', 'ctrl_d', 'ctrl_c', 'clear', 'ctrl_k', 'ctrl_u'

    char: required when key == 'char'

    Returns (submit, exit_signal) — exit True means Ctrl-D/Ctrl-C.

    Mutates buf.
    """
    if key == "char":
        if char:
            buf.insert(char)
        return (False, False)
    elif key == "backspace":
        buf.backspace()
        return (False, False)
    elif key == "delete":
        buf.delete()
        return (False, False)
    elif key == "left":
        buf.move_left()
        return (False, False)
    elif key == "right":
        buf.move_right()
        return (False, False)
    elif key == "home":
        buf.move_home()
        return (False, False)
    elif key == "end":
        buf.move_end()
        return (False, False)
    elif key == "clear":
        buf.clear()
        return (False, False)
    elif key == "ctrl_k":
        # delete to end
        buf.text = buf.text[: buf.cursor]
        return (False, False)
    elif key == "ctrl_u":
        buf.clear()
        return (False, False)
    elif key == "enter":
        return (True, False)
    elif key == "ctrl_d":
        return (False, True)
    elif key == "ctrl_c":
        return (False, True)
    else:
        # unknown key — ignore
        return (False, False)


def _arg_hint(name: str, args_json: str) -> str:
    """Pure one-line hint for batch summaries: path/query, else ''."""
    try:
        args = json.loads(args_json) if args_json else {}
    except Exception:
        return ""
    if name in ("read_file", "edit_file", "write_file"):
        return str(args.get("path", ""))
    if name == "web_search":
        return str(args.get("query", ""))
    return ""


def _diff_kind(line: str) -> str:
    """Pure diff-line classifier for coloring."""
    s = line.lstrip()
    if s.startswith("+++") or s.startswith("---"):
        return K_DIM
    if s.startswith("@@"):
        return K_HUNK
    if s.startswith("+"):
        return K_ADD
    if s.startswith("-"):
        return K_DEL
    return K_DIM


# ----------------------------------------------------------------------
# TuiRenderer — curses dependent part, but keeps render-state in pure helpers
# ----------------------------------------------------------------------

class TuiRenderer:
    """Implements ui.Renderer protocol with curses TUI.

    Usage:
        renderer = TuiRenderer(model_name="...", session_id="...")
        renderer.start_curses()
        try:
            # drive calls, read_line loops
        finally:
            renderer.stop_curses()

    When not in curses mode, it still accumulates transcript (useful for tests).
    """

    def __init__(self, model_name: str = "unknown", session_id: str = "", max_lines: int = 2000):
        self.model_name: str = model_name or "unknown"
        self.session_id: str = session_id or ""
        self.step_count: int = 0
        self.transcript = TranscriptBuffer(max_lines=max_lines)

        # curses state (None when not active)
        self._stdscr = None
        self._transcript_win = None
        self._status_win = None
        self._input_win = None
        self._scroll_offset: int = 0
        self._in_curses: bool = False
        self._width: int = 80
        self._height: int = 24
        self._transcript_height: int = max(1, 24 - 2)
        self._did_alt_screen: bool = False
        self._batch: list[dict] | None = None
        self._batch_start: int = 0
        self._attrs: dict[str, int] = {}  # kind -> curses attr, filled in start_curses

    # ---- pure state update ----

    def update_status(self, model_name: str | None = None, step_count: int | None = None, session_id: str | None = None) -> None:
        if model_name is not None:
            self.model_name = model_name or "unknown"
        if step_count is not None:
            try:
                self.step_count = int(step_count)
            except Exception:
                pass
        if session_id is not None:
            self.session_id = session_id or ""
        self._redraw_status_if_needed()

    # ---- Renderer protocol ----

    def begin_tools(self, step_num: int) -> None:
        self.step_count = step_num
        self._batch = []
        self._batch_start = len(self.transcript)
        self._redraw_transcript_if_needed()
        self._redraw_status_if_needed()

    def end_tools(self) -> None:
        batch = self._batch
        self._batch = None
        if not batch:
            return
        text, has_error = summarize_batch([(e["name"], e["hint"], e["ok"]) for e in batch])
        marker = "✖" if has_error else "✔"
        kind = K_SUMMARY_ERR if has_error else K_SUMMARY
        self.transcript.collapse_from(self._batch_start, [(kind, f"{marker} ↳ {text}")])
        self._redraw_transcript_if_needed()

    def step(self, num: int, total: int, text: str) -> None:
        self.step_count = num
        truncated = (text or "")[:500].replace("\n", " ")
        self.transcript.append(f"Step {num}/{total} {truncated}", K_STEP)
        self._redraw_transcript_if_needed()
        self._redraw_status_if_needed()

    def tool_call(self, num: int, name: str, args_json: str) -> None:
        truncated = (args_json or "")[:200].replace("\n", " ")
        self.transcript.append(f"  tool: {name} {truncated}", K_TOOL)
        if self._batch is not None:
            self._batch.append({"name": name, "hint": _arg_hint(name, args_json), "ok": True, "_open": True})
        self._redraw_transcript_if_needed()

    def tool_result(self, tool_name: str, text: str) -> None:
        raw = text or ""
        is_error = raw.lstrip().startswith("error:")
        if self._batch is not None:
            for entry in reversed(self._batch):
                if entry["name"] == tool_name and entry["_open"]:
                    entry["_open"] = False
                    entry["ok"] = not is_error
                    break
        kind = K_ERR if is_error else K_OK
        first = raw.splitlines()[0] if raw.splitlines() else ""
        self.transcript.append(f"  -> {tool_name}: {first[:500]}", kind)
        # a few extra diff lines for edits/writes
        if tool_name in ("edit_file", "write_file") and not is_error:
            for ln in raw.splitlines()[1:6]:
                self.transcript.append(f"     {ln[:800]}", _diff_kind(ln))
        self._redraw_transcript_if_needed()

    def final(self, summary: str) -> None:
        self.transcript.append(f"result: {summary}", K_RESULT)
        self._redraw_transcript_if_needed()

    # ---- alternate screen helpers ----

    def _enter_alt_screen(self) -> None:
        try:
            sys.stdout.write("\x1b[?1049h")
            sys.stdout.flush()
            self._did_alt_screen = True
        except Exception:
            self._did_alt_screen = False

    def _exit_alt_screen(self) -> None:
        if not self._did_alt_screen:
            return
        try:
            sys.stdout.write("\x1b[?1049l")
            sys.stdout.flush()
        except Exception:
            pass
        self._did_alt_screen = False

    # ---- curses lifecycle ----

    def start_curses(self) -> None:
        """Enter curses, alternate screen, setup windows."""
        try:
            import curses  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"curses not available: {e}")
        import curses

        self._enter_alt_screen()
        try:
            self._stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self._stdscr.keypad(True)
            try:
                curses.curs_set(1)
            except curses.error:
                pass
            self._init_colors()
            self._in_curses = True
            self._setup_windows()
            self._redraw_all()
        except Exception:
            # try cleanup
            try:
                if self._stdscr is not None:
                    self._stdscr.keypad(False)
                    curses.nocbreak()
                    curses.echo()
                    curses.endwin()
            except Exception:
                pass
            self._stdscr = None
            self._in_curses = False
            self._exit_alt_screen()
            raise

    def stop_curses(self) -> None:
        """Leave curses and restore terminal."""
        try:
            import curses
        except Exception:
            # no curses, just exit alt screen
            self._exit_alt_screen()
            return

        if self._stdscr is not None:
            try:
                self._stdscr.keypad(False)
                curses.nocbreak()
                curses.echo()
                curses.endwin()
            except Exception:
                pass
        self._stdscr = None
        self._transcript_win = None
        self._status_win = None
        self._input_win = None
        self._in_curses = False
        self._exit_alt_screen()

    # Context manager support
    def __enter__(self):
        self.start_curses()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_curses()
        return False

    # ---- window setup / drawing (curses part) ----

    def _setup_windows(self) -> None:
        if self._stdscr is None:
            return
        try:
            import curses
        except Exception:
            return
        try:
            h, w = self._stdscr.getmaxyx()
        except curses.error:
            h, w = self._height, self._width
        self._height = h
        self._width = w
        status_h = 1
        input_h = INPUT_ROWS
        transcript_h = max(1, h - status_h - input_h)
        self._transcript_height = transcript_h
        try:
            self._transcript_win = curses.newwin(transcript_h, w, 0, 0)
            self._status_win = curses.newwin(status_h, w, transcript_h, 0)
            self._input_win = curses.newwin(input_h, w, transcript_h + status_h, 0)
            # allow keypad on input win as well
            self._transcript_win.keypad(True)
            self._status_win.keypad(True)
            self._input_win.keypad(True)
        except curses.error:
            self._transcript_win = None
            self._status_win = None
            self._input_win = None

    def _redraw_all(self) -> None:
        self._draw_transcript()
        self._draw_status()
        # input will be drawn by read_line loop; but clear it
        if self._input_win is not None:
            try:
                self._input_win.erase()
                self._input_win.noutrefresh()
            except Exception:
                pass
        try:
            import curses
            curses.doupdate()
        except Exception:
            pass

    def _draw_transcript(self) -> None:
        if not self._in_curses or self._stdscr is None or self._transcript_win is None:
            return
        try:
            import curses
            self._transcript_win.erase()
            height = self._transcript_height
            width = max(8, self._width)
            # wrap first, then take the bottom window (wrapped continuation counts as lines)
            wrapped: list[tuple[str, str]] = []
            for kind, line in self.transcript.items():
                for chunk in wrap_line(line, width - 1):
                    wrapped.append((kind, chunk))
            if self._scroll_offset:
                total = len(wrapped)
                max_offset = max(0, total - height)
                off = min(max(0, self._scroll_offset), max_offset)
                start = max(0, total - height - off)
                vis = wrapped[start:start + height]
            else:
                vis = wrapped[max(0, len(wrapped) - height):]
            for idx, (kind, chunk) in enumerate(vis):
                if idx >= height:
                    break
                try:
                    if kind in (K_SUMMARY, K_SUMMARY_ERR) and len(chunk) >= 2:
                        marker_attr = self._kind_attr(K_ERR if kind == K_SUMMARY_ERR else K_OK)
                        self._transcript_win.addnstr(idx, 0, chunk[:2], width - 1, marker_attr)
                        self._transcript_win.addnstr(idx, 2, chunk[2:], max(0, width - 3), self._kind_attr(kind))
                    else:
                        self._transcript_win.addnstr(idx, 0, chunk, width - 1, self._kind_attr(kind))
                except curses.error:
                    pass
            self._transcript_win.noutrefresh()
            curses.doupdate()
        except Exception:
            pass

    def _draw_status(self) -> None:
        if not self._in_curses or self._status_win is None:
            return
        try:
            import curses
            self._status_win.erase()
            status_str = format_status(self.model_name, self.step_count, self.session_id)
            truncated = status_str[: max(0, self._width - 1)]
            try:
                # reverse attribute if available
                attr = curses.A_REVERSE if hasattr(curses, "A_REVERSE") else 0
                self._status_win.addnstr(0, 0, truncated, max(0, self._width - 1), attr)
            except curses.error:
                pass
            self._status_win.noutrefresh()
            curses.doupdate()
        except Exception:
            pass

    def _draw_input(self, input_buf: InputBuffer, prompt: str = "> ") -> None:
        if not self._in_curses or self._input_win is None:
            return
        try:
            import curses
            self._input_win.erase()
            rows, crow, ccol = wrap_input(prompt, input_buf.text, input_buf.cursor, self._width, INPUT_ROWS)
            for i, row in enumerate(rows):
                if i >= INPUT_ROWS:
                    break
                self._input_win.addnstr(i, 0, row, max(0, self._width - 1))
            try:
                self._input_win.move(
                    max(0, min(crow, INPUT_ROWS - 1)),
                    max(0, min(ccol, self._width - 1)),
                )
            except curses.error:
                pass
            self._input_win.noutrefresh()
            curses.doupdate()
        except Exception:
            pass

    def _clear_input(self, prompt: str = "> ") -> None:
        """Erase the input window after submit so stale text never lingers."""
        if not self._in_curses or self._input_win is None:
            return
        try:
            self._input_win.erase()
            self._input_win.noutrefresh()
            import curses
            curses.doupdate()
        except Exception:
            pass

    def _init_colors(self) -> None:
        """Set up kind -> attr map. Falls back to bold/dim when colors fail."""
        self._attrs = {}
        try:
            import curses
            curses.start_color()
            try:
                curses.use_default_colors()
            except curses.error:
                pass
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_RED, -1)
            curses.init_pair(3, curses.COLOR_CYAN, -1)
            green = curses.color_pair(1)
            red = curses.color_pair(2)
            cyan = curses.color_pair(3)
        except Exception:
            green = red = cyan = 0
        try:
            import curses
            bold, dim = curses.A_BOLD, curses.A_DIM
        except Exception:
            bold, dim = 0, 0
        self._attrs = {
            K_STEP: bold,
            K_TOOL: cyan,
            K_OK: green,
            K_ERR: red,
            K_ADD: green,
            K_DEL: red,
            K_HUNK: cyan,
            K_DIM: dim,
            K_INPUT: bold,
            K_RESULT: bold,
            K_SUMMARY: dim,
            K_SUMMARY_ERR: dim,
            K_PLAIN: 0,
        }

    def _kind_attr(self, kind: str) -> int:
        return self._attrs.get(kind, 0)

    # ---- helpers for redraw checks ----
    def _redraw_transcript_if_needed(self) -> None:
        if self._in_curses:
            # if currently at bottom, keep auto-scroll
            # we keep _scroll_offset = 0 on new output to stay at bottom
            # unless user scrolled up, we could preserve? But for simplicity reset if at bottom-ish.
            # We'll reset to 0 if we were at bottom before (0) else keep.
            # Here we just ensure if offset==0 we stay, else we keep.
            self._draw_transcript()

    def _redraw_status_if_needed(self) -> None:
        if self._in_curses:
            self._draw_status()

    # ---- input loop ----

    def read_line(self, prompt: str = "> ") -> str | None:
        """Read a single line with curses editing if in curses mode, else fallback to input().

        Returns None on Ctrl-D / Ctrl-C / EOF to signal exit.
        """
        if not self._in_curses or self._stdscr is None:
            try:
                return input(prompt)
            except EOFError:
                return None
            except KeyboardInterrupt:
                return None

        import curses

        input_buf = InputBuffer()
        self._setup_windows()  # ensure size up to date

        while True:
            self._draw_transcript()
            self._draw_status()
            self._draw_input(input_buf, prompt)

            try:
                # Try wide char version first (supports unicode)
                try:
                    # get_wch may raise if no input; attempt it
                    wch = self._stdscr.get_wch()
                except AttributeError:
                    # fallback to getch
                    wch = self._stdscr.getch()
                except curses.error:
                    continue
            except Exception:
                continue

            # wch can be str or int
            if isinstance(wch, str):
                # String keys: check special
                if wch == "\n" or wch == "\r":
                    line = input_buf.text
                    self.transcript.append(f"> {line}", K_INPUT)
                    self._scroll_offset = 0
                    self._draw_transcript()
                    self._clear_input(prompt)
                    return line
                if wch == "\x04":  # Ctrl-D
                    return None
                if wch == "\x03":  # Ctrl-C
                    return None
                if wch == "\x7f" or wch == "\b":  # backspace
                    input_buf.backspace()
                    continue
                if wch == "\x01":  # Ctrl-A
                    input_buf.move_home()
                    continue
                if wch == "\x05":  # Ctrl-E
                    input_buf.move_end()
                    continue
                if wch == "\x0b":  # Ctrl-K
                    input_buf.text = input_buf.text[: input_buf.cursor]
                    continue
                if wch == "\x15":  # Ctrl-U
                    input_buf.clear()
                    continue
                # Printable char
                if wch.isprintable():
                    input_buf.insert(wch)
                continue
            else:
                # int keycode
                key = wch
                if key in (curses.KEY_ENTER, 10, 13):
                    line = input_buf.text
                    self.transcript.append(f"> {line}", K_INPUT)
                    self._scroll_offset = 0
                    self._clear_input(prompt)
                    return line
                if key == 4:  # Ctrl-D
                    return None
                if key == 3:  # Ctrl-C
                    return None
                if key in (curses.KEY_BACKSPACE, 127, 8):
                    input_buf.backspace()
                    continue
                if key == curses.KEY_DC:
                    input_buf.delete()
                    continue
                if key == curses.KEY_LEFT:
                    input_buf.move_left()
                    continue
                if key == curses.KEY_RIGHT:
                    input_buf.move_right()
                    continue
                if key == curses.KEY_HOME:
                    input_buf.move_home()
                    continue
                if key == curses.KEY_END:
                    input_buf.move_end()
                    continue
                if key == curses.KEY_RESIZE:
                    self._setup_windows()
                    continue
                if key == curses.KEY_PPAGE:
                    max_off = max(0, len(self.transcript) - self._transcript_height)
                    self._scroll_offset = min(max_off, self._scroll_offset + max(1, self._transcript_height // 2))
                    continue
                if key == curses.KEY_NPAGE:
                    self._scroll_offset = max(0, self._scroll_offset - max(1, self._transcript_height // 2))
                    continue
                # Ctrl-A, Ctrl-E as ints as well (1,5)
                if key == 1:
                    input_buf.move_home()
                    continue
                if key == 5:
                    input_buf.move_end()
                    continue
                if key == 11:  # Ctrl-K
                    input_buf.text = input_buf.text[: input_buf.cursor]
                    continue
                if key == 21:  # Ctrl-U
                    input_buf.clear()
                    continue
                # Regular ASCII
                if 32 <= key <= 126:
                    try:
                        ch = chr(key)
                        input_buf.insert(ch)
                    except Exception:
                        pass
                    continue
                if key >= 160:
                    try:
                        ch = chr(key)
                        if ch.isprintable():
                            input_buf.insert(ch)
                    except Exception:
                        pass
                    continue
                # otherwise ignore
                continue
