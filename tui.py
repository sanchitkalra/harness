"""Textual TUI for mini coding-agent.

TuiRenderer is a Textual App that also implements the ui.Renderer protocol
(step / tool_call / tool_result / begin_tools / end_tools / final), plus
read_line() for input and update_status()/cancelled() used by agent.py.

Renderer methods are called from the interactive-session worker thread
(see agent.py main()); they marshal onto the Textual UI thread with
call_from_thread before touching any widget.

Pure helpers (summarize_batch, format_file_diff, _arg_hint) stay
framework-free so they're covered by plain unit tests.
"""
from __future__ import annotations

import json
import queue
import re
import threading

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Input, Static

# ----------------------------------------------------------------------
# Pure helpers (no Textual/UI imports) — reused by TuiRenderer.
# ----------------------------------------------------------------------


def format_status(model_name: str, step_count: int | None = None, session_id: str | None = None) -> str:
    model = str(model_name or "unknown").strip() or "unknown"
    steps = 0
    if step_count is not None:
        try:
            steps = int(step_count)
        except Exception:
            steps = 0
    sid = (session_id or "").strip()
    return f"{model} | steps {steps}" + (f" | {sid}" if sid else "")


def _arg_hint(name: str, args_json: str) -> str:
    try:
        args = json.loads(args_json) if args_json else {}
    except Exception:
        return ""
    if name in ("read_file", "edit_file", "write_file"):
        return str(args.get("path", ""))
    if name == "web_search":
        return str(args.get("query", ""))
    return ""


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def summarize_batch(entries: list[tuple[str, str, bool]]) -> tuple[str, bool]:
    """Pure per-step summary. entries = (tool_name, arg_hint, ok)."""
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


def format_file_diff(diff_lines: list[str], path_hint: str = "") -> tuple[str, str, list[tuple[str, str]]]:
    """Pure Claude-style file-diff formatter.

    Returns (filename, stat_line, styled_lines) where styled_lines are
    (kind, text) with gutter numbers; kind is one of "add"/"del"/"dim".
    """
    filename = path_hint
    hunks: list[str] = []
    for ln in diff_lines:
        if ln.startswith("+++ "):
            filename = ln[4:].strip()
            if filename.startswith("b/"):
                filename = filename[2:]
        elif ln.startswith("--- ") or ln.startswith("+++"):
            continue
        else:
            hunks.append(ln)

    styled: list[tuple[str, str]] = []
    added = removed = 0
    old = new = 0
    in_hunk = False
    for ln in hunks:
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", ln)
        if m:
            old, new = int(m.group(1)), int(m.group(2))
            in_hunk = True
            continue
        if not in_hunk:
            if ln.startswith("+") and not ln.startswith("+++"):
                added += 1
                new += 1
                styled.append(("add", f"{new:>4} + {ln[1:]}"))
            elif ln.startswith("-") and not ln.startswith("---"):
                removed += 1
                old += 1
                styled.append(("del", f"{old:>4} - {ln[1:]}"))
            elif ln:
                styled.append(("dim", f"      {ln}"))
            continue
        if ln.startswith("+"):
            added += 1
            styled.append(("add", f"{new:>4} + {ln[1:]}"))
            new += 1
        elif ln.startswith("-"):
            removed += 1
            styled.append(("del", f"{old:>4} - {ln[1:]}"))
            old += 1
        elif ln.startswith(" "):
            styled.append(("dim", f"{old:>4}   {ln[1:]}"))
            old += 1
            new += 1
        elif ln.startswith("\\"):
            continue  # "\ No newline at end of file"

    if added and not removed:
        stat = f"Added {_plural(added, 'line')}"
    elif removed and not added:
        stat = f"Removed {_plural(removed, 'line')}"
    else:
        stat = f"Added {_plural(added, 'line')}, removed {_plural(removed, 'line')}"
    return filename, stat, styled


# ----------------------------------------------------------------------
# TuiRenderer — Textual App + Renderer protocol
# ----------------------------------------------------------------------

class TuiRenderer(App):
    """Textual app. Renderer-protocol methods run on a worker thread and
    marshal onto the UI thread via call_from_thread."""

    CSS = """
    #status { background: $primary-darken-1; color: $text; height: 1; }
    #transcript { padding: 0 1; }
    .step { text-style: bold; }
    .tool { color: $text-muted; }
    .ok { color: $success; }
    .err { color: $error; text-style: bold; }
    .head { text-style: bold; }
    .dim { color: $text-muted; }
    .add { background: $success-muted; color: $text; }
    .del { background: $error-muted; color: $text; }
    .result { text-style: bold; }
    .usermsg { border: round $accent; padding: 0 1; margin: 0 0; }
    Collapsible { padding: 0; margin: 0; }
    Collapsible.err > CollapsibleTitle { color: $error; }
    Collapsible.ok > CollapsibleTitle { color: $success; }
    """
    # ponytail: quit is only reachable between turns (queue-driven), not while
    # a turn is running on the worker thread — Python can't deliver a signal
    # into another thread. Upgrade path: cooperative cancellation flag checked
    # between tool calls in agent.drive(), if killing a running turn matters.
    BINDINGS = [("ctrl+c", "quit_app", "Quit"), ("ctrl+d", "quit_app", "Quit")]

    def __init__(self, model_name: str = "unknown", session_id: str = ""):
        super().__init__()
        self.model_name = model_name or "unknown"
        self.session_id = session_id or ""
        self.step_count = 0
        self._input_queue: "queue.Queue[str | None]" = queue.Queue()
        self._batch: list[dict] | None = None
        self._transcript: VerticalScroll | None = None
        self._status: Static | None = None
        self._input: Input | None = None
        # Set once the app's event loop is actually running — call_from_thread
        # raises if invoked before this, so the session worker thread must
        # wait on it before touching any Renderer method.
        self._loop_ready = threading.Event()

    def wait_until_ready(self, timeout: float | None = None) -> None:
        self._loop_ready.wait(timeout)

    # ---- Textual app ----

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        yield VerticalScroll(id="transcript")
        yield Input(placeholder="type a task, enter to send", id="input")

    def on_mount(self) -> None:
        self._transcript = self.query_one("#transcript", VerticalScroll)
        self._status = self.query_one("#status", Static)
        self._input = self.query_one("#input", Input)
        self._refresh_status()
        self._input.focus()
        self._loop_ready.set()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value
        self._input.value = ""
        self._input_queue.put(text)

    def action_quit_app(self) -> None:
        self._input_queue.put(None)
        self.exit()

    def _refresh_status(self) -> None:
        if self._status is not None:
            self._status.update(Text(format_status(self.model_name, self.step_count, self.session_id)))

    def _mount_line(self, text: str, cls: str = "") -> Static:
        w = Static(Text(text), classes=cls)
        self._transcript.mount(w)
        self._transcript.scroll_end(animate=False)
        return w

    def _mount_user_message(self, text: str) -> None:
        self._transcript.mount(Static(Text(text), classes="usermsg"))
        self._transcript.scroll_end(animate=False)

    def _set_input_enabled(self, enabled: bool) -> None:
        if self._input is not None:
            self._input.disabled = not enabled
            if enabled:
                self._input.focus()

    # ---- Renderer protocol (called from the session worker thread) ----

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
        self.call_from_thread(self._refresh_status)

    def begin_tools(self, step_num: int) -> None:
        self.step_count = step_num
        self._batch = []
        self.call_from_thread(self._refresh_status)

    def tool_call(self, num: int, name: str, args_json: str) -> None:
        truncated = (args_json or "")[:200].replace("\n", " ")
        widget = self.call_from_thread(self._mount_line, f"  tool: {name} {truncated}", "tool")
        if self._batch is not None:
            self._batch.append({
                "name": name,
                "hint": _arg_hint(name, args_json),
                "ok": True,
                "_open": True,
                "widgets": [widget],
                "details": [],
            })

    def tool_result(self, tool_name: str, text: str) -> None:
        raw = text or ""
        is_error = raw.lstrip().startswith("error:")
        matched = None
        if self._batch is not None:
            for entry in reversed(self._batch):
                if entry["name"] == tool_name and entry["_open"]:
                    entry["_open"] = False
                    entry["ok"] = not is_error
                    matched = entry
                    break
        kind = "err" if is_error else "ok"
        new_widgets: list[Static] = []
        new_details: list[tuple[str, str]] = []
        if tool_name in ("edit_file", "write_file") and not is_error:
            first_line = raw.splitlines()[0] if raw.splitlines() else ""
            m = re.match(r"ok: (?:edited|wrote|overwrote) (\S+)", first_line)
            path_hint = m.group(1) if m else ""
            fname, stat, styled = format_file_diff(raw.splitlines()[1:], path_hint)
            head_text = f"Update({fname})" if fname else "Update"
            new_widgets.append(self.call_from_thread(self._mount_line, head_text, "head"))
            new_details.append(("head", head_text))
            new_widgets.append(self.call_from_thread(self._mount_line, f"└ {stat}", "dim"))
            new_details.append(("dim", f"└ {stat}"))
            for k, t in styled:
                new_widgets.append(self.call_from_thread(self._mount_line, t, k))
                new_details.append((k, t))
        else:
            first = raw.splitlines()[0] if raw.splitlines() else ""
            line = f"  -> {tool_name}: {first[:500]}"
            new_widgets.append(self.call_from_thread(self._mount_line, line, kind))
            new_details.append((kind, line))
        if matched is not None:
            matched["widgets"].extend(new_widgets)
            matched["details"].extend(new_details)

    def end_tools(self) -> None:
        batch = self._batch
        self._batch = None
        if not batch:
            return
        is_edit = lambda e: e["name"] in ("edit_file", "write_file")
        others = [e for e in batch if not is_edit(e) and e["name"] != "done"]

        def finish() -> None:
            for e in batch:
                if not is_edit(e):
                    for w in e["widgets"]:
                        w.remove()
            if others:
                text, has_error = summarize_batch([(e["name"], e["hint"], e["ok"]) for e in others])
                marker = "✖" if has_error else "✔"
                detail_widgets = []
                for e in others:
                    for kind, dtext in e["details"]:
                        detail_widgets.append(Static(Text(dtext), classes=kind))
                collapsible = Collapsible(
                    *detail_widgets,
                    title=f"{marker} {text}",
                    collapsed=True,
                    classes="err" if has_error else "ok",
                )
                self._transcript.mount(collapsible)
                self._transcript.scroll_end(animate=False)

        self.call_from_thread(finish)

    def step(self, num: int, total: int, text: str) -> None:
        self.step_count = num
        self.call_from_thread(self._mount_line, f"Step {num}/{total} {(text or '').strip()}", "step")
        self.call_from_thread(self._refresh_status)

    def final(self, summary: str) -> None:
        self.call_from_thread(self._mount_line, f"result: {summary}", "result")

    def cancelled(self) -> None:
        self.call_from_thread(self._mount_line, "[cancelled]", "err")

    # ---- input (called from the session worker thread) ----

    def read_line(self, prompt: str = "> ") -> str | None:
        self.call_from_thread(self._set_input_enabled, True)
        line = self._input_queue.get()
        self.call_from_thread(self._set_input_enabled, False)
        if line is None:
            return None
        self.call_from_thread(self._mount_user_message, line)
        return line
