"""Curses TUI for mini coding-agent — stdlib curses only, macOS/Linux.

Provides:
- pure helpers: TranscriptBuffer, InputBuffer, format_status, visible_slice helpers, handle_input_event
- TuiRenderer: implements ui.Renderer protocol with alternate-screen, scrollable transcript, single-line input, status bar

Curses calls are isolated in TuiRenderer methods; pure helpers are terminal-free.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable

# ----------------------------------------------------------------------
# Pure helpers (no curses)
# ----------------------------------------------------------------------

class TranscriptBuffer:
    """Bounded transcript buffer — keeps last max_lines lines."""

    def __init__(self, max_lines: int = 2000):
        if max_lines <= 0:
            max_lines = 2000
        self.max_lines = max_lines
        self._lines: list[str] = []

    def append(self, text: str) -> None:
        """Append text (may be multiline) as separate lines, then cap."""
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        parts = text.splitlines() or [""]
        for p in parts:
            self._lines.append(p)
        # cap to max_lines (keep newest)
        if len(self._lines) > self.max_lines:
            self._lines = self._lines[-self.max_lines :]

    def extend(self, lines: Iterable[str]) -> None:
        for ln in lines:
            self.append(ln)

    def clear(self) -> None:
        self._lines.clear()

    @property
    def lines(self) -> list[str]:
        return list(self._lines)

    def __len__(self) -> int:
        return len(self._lines)

    def get_visible(self, height: int, scroll_offset: int = 0) -> list[str]:
        """Return visible slice for a window of given height.
        scroll_offset = 0 means bottom is visible (auto-scroll). Positive means scrolled up.
        """
        return visible_slice(self._lines, height, scroll_offset)


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


def visible_input(prompt: str, text: str, cursor: int, width: int) -> tuple[str, int]:
    """Pure horizontal-scroll helper for the single-line input box.

    Returns (display, cursor_x): display fits in width, cursor_x is the
    cursor column within display. The view scrolls so the cursor stays
    visible; the tail is shown when the cursor is at the end.
    """
    prompt = prompt or ""
    text = text or ""
    cursor = max(0, min(cursor, len(text)))
    width = max(1, width)
    avail = max(1, width - len(prompt) - 1)
    if len(text) <= avail:
        start = 0
    elif cursor < avail:
        start = 0
    elif cursor >= len(text):
        start = len(text) - avail
    else:
        start = cursor - avail + 1
    start = max(0, start)
    return prompt + text[start:start + avail], len(prompt) + (cursor - start)


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
        # Optional marker; keep light to not clutter
        self.transcript.append(f"[{step_num}] tools…")
        self._redraw_transcript_if_needed()
        self._redraw_status_if_needed()

    def end_tools(self) -> None:
        # no-op, but could redraw
        pass

    def step(self, num: int, total: int, text: str) -> None:
        self.step_count = num
        truncated = (text or "")[:500].replace("\n", " ")
        self.transcript.append(f"Step {num}/{total} {truncated}")
        self._redraw_transcript_if_needed()
        self._redraw_status_if_needed()

    def tool_call(self, num: int, name: str, args_json: str) -> None:
        truncated = (args_json or "")[:200].replace("\n", " ")
        self.transcript.append(f"  tool: {name} {truncated}")
        self._redraw_transcript_if_needed()

    def tool_result(self, tool_name: str, text: str) -> None:
        raw = text or ""
        first = raw.splitlines()[0] if raw.splitlines() else ""
        first = first[:500]
        self.transcript.append(f"  -> {tool_name}: {first}")
        # if multiline diff-like, show up to few lines
        if tool_name in ("edit_file", "write_file"):
            lines = raw.splitlines()[1:6]  # show a few extra
            for ln in lines:
                self.transcript.append(f"     {ln[:800]}")
        self._redraw_transcript_if_needed()

    def final(self, summary: str) -> None:
        self.transcript.append(f"result: {summary}")
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
        input_h = 1
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
            width = self._width
            vis = self.transcript.get_visible(height, self._scroll_offset)
            for idx, line in enumerate(vis):
                if idx >= height:
                    break
                truncated = line[: max(0, width - 1)]
                try:
                    self._transcript_win.addnstr(idx, 0, truncated, max(0, width - 1))
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
            display, cursor_x = visible_input(prompt, input_buf.text, input_buf.cursor, self._width)
            self._input_win.addnstr(0, 0, display, max(0, self._width - 1))
            try:
                self._input_win.move(0, max(0, min(cursor_x, self._width - 1)))
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
                    self.transcript.append(f"{prompt}{line}")
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
                    self.transcript.append(f"{prompt}{line}")
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
