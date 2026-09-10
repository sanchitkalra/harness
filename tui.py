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
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Select, Static, TextArea

import model_registry

# ----------------------------------------------------------------------
# Pure helpers (no Textual/UI imports) — reused by TuiRenderer.
# ----------------------------------------------------------------------


def format_status(step_count: int, step_total: int | None = None) -> str:
    try:
        steps = int(step_count)
    except Exception:
        steps = 0
    if step_total:
        return f"step {steps}/{int(step_total)}"
    return f"step {steps}"


def format_footer(session_name: str, session_id: str, group_name: str, model_name: str) -> tuple[str, str]:
    """(session line, group/model line) shown under the input box."""
    session_line = (session_name or "").strip() or (session_id or "").strip() or "(unnamed session)"
    group = (group_name or "").strip()
    model = str(model_name or "unknown").strip() or "unknown"
    model_line = f"{group}/{model}" if group else model
    return session_line, model_line


def _arg_hint(name: str, args_json: str) -> str:
    try:
        args = json.loads(args_json) if args_json else {}
    except Exception:
        return ""
    if name in ("read_file", "edit_file", "write_file"):
        return str(args.get("path", ""))
    if name == "web_search":
        return str(args.get("query", ""))
    if name == "bash":
        return str(args.get("command", ""))
    return ""


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def summarize_batch(entries: list[tuple[str, str, bool]]) -> tuple[str, bool]:
    """Pure per-step summary. entries = (tool_name, arg_hint, ok)."""
    reads: list[str] = []
    edits: list[str] = []
    writes: list[str] = []
    cmds: list[str] = []
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
            cmds.append(hint or "?")
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
    if cmds:
        if len(cmds) == 1:
            parts.append(f"ran: {cmds[0]}"[:80])
        else:
            uniq = list(dict.fromkeys(cmds))[:3]
            more = f" +{len(cmds) - len(uniq)} more" if len(cmds) > 3 else ""
            parts.append(f"ran {len(cmds)} commands: {', '.join(uniq)}{more}"[:120])
    if searches:
        parts.append(f"searched {searches[0]!r}"[:80] if len(searches) == 1 else f"searched {len(searches)} queries")
    for k, v in others.items():
        parts.append(k if v == 1 else f"{k} x{v}")
    if errors:
        parts.append(f"{errors} error(s)")
    return (" · ".join(parts)) or "tools", errors > 0


RUNNABLE_KINDS = ("read_file", "bash")  # tool kinds that collapse into one running summary


def run_title(kind: str, hints: list[str], has_error: bool) -> str:
    """Title for a cross-step collapsed run of same-kind tool calls
    (repeated read_file or bash calls). Pure so it's unit-testable."""
    marker = "✖" if has_error else "✔"
    n = len(hints)
    if kind == "read_file":
        uniq = list(dict.fromkeys(hints))
        if len(uniq) == 1:
            body = uniq[0] if n == 1 else f"{uniq[0]} ({_plural(n, 'read')})"
            body = f"read {body}"
        else:
            shown = uniq[:4]
            more = f" +{len(uniq) - len(shown)} more" if len(uniq) > len(shown) else ""
            body = f"read {n} files: {', '.join(shown)}{more}"
    else:  # bash
        body = f"ran {_plural(n, 'command')}"
    return f"{marker} {body}"


SLASH_COMMANDS: dict[str, str] = {
    "help": "list available commands",
    "name": "rename this session (shown in the status bar)",
    "rename": "alias for /name",
    "clear": "clear the transcript and start a new conversation",
    "login": "pick or add a login (provider + API key)",
    "model": "pick or add a model within the current login",
    "quit": "exit the app",
    "exit": "alias for /quit",
}


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
# LoginPickerScreen — /login: pick a saved auth group, or add a new one
# ----------------------------------------------------------------------

_PICKER_CSS = """
ModalScreen { align: center middle; }
#picker-box { width: 64; height: auto; border: round $accent; padding: 1 2; background: $panel; }
#picker-box Input, #picker-box Select { margin-bottom: 1; }
#add-form { display: none; }
#add-form.visible { display: block; }
.picker-err { color: $error; }
"""


class LoginPickerScreen(ModalScreen[str | None]):
    """Dismisses with the newly-active group name, or None if cancelled."""

    CSS = _PICKER_CSS
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        reg = model_registry.load_registry()
        self._groups = reg["groups"]
        self._active = reg.get("active_group")

    def compose(self) -> ComposeResult:
        options = [
            (
                f"{name} — {g.get('provider')} ({model_registry.mask_key(g.get('api_key', ''))})"
                + (" (current)" if name == self._active else ""),
                name,
            )
            for name, g in self._groups.items()
        ]
        options.append(("+ Add new login...", "__add__"))
        with Vertical(id="picker-box"):
            yield Static("Logins — enter to select, esc to cancel")
            yield Select(options, id="picker-select", allow_blank=True)
            with Vertical(id="add-form"):
                yield Input(placeholder="name (e.g. work)", id="f-name")
                yield Select(
                    [(p, p) for p in model_registry.PROVIDERS], prompt="provider", id="f-provider"
                )
                yield Input(placeholder="API key", password=True, id="f-key")
                yield Input(placeholder="model (e.g. claude-sonnet-5)", id="f-model")
                yield Input(placeholder="base url (required for compatible)", id="f-base")
                yield Button("Save", id="f-save", variant="primary")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "picker-select":
            return
        if event.value == "__add__":
            self.query_one("#add-form").add_class("visible")
            self.query_one("#f-name", Input).focus()
        elif event.value is not Select.BLANK:
            model_registry.set_active_group(event.value)
            self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "f-save":
            return
        name = self.query_one("#f-name", Input).value.strip()
        provider = self.query_one("#f-provider", Select).value
        api_key = self.query_one("#f-key", Input).value.strip()
        model_name = self.query_one("#f-model", Input).value.strip()
        base_url = self.query_one("#f-base", Input).value.strip() or None
        box = self.query_one("#picker-box")
        for old in box.query(".picker-err"):
            old.remove()
        if not name or provider is Select.BLANK or not api_key or not model_name:
            box.mount(Static("name, provider, API key, and model are required", classes="picker-err"))
            return
        if provider == "compatible" and not base_url:
            box.mount(Static("base url is required for a compatible provider", classes="picker-err"))
            return
        model_registry.add_group(name, provider, api_key, model_name, base_url)
        self.dismiss(name)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ----------------------------------------------------------------------
# ModelPickerScreen — /model: pick a model within the active login, or
# add a new one to it. Requires an active login (use /login first).
# ----------------------------------------------------------------------

class ModelPickerScreen(ModalScreen[str | None]):
    """Dismisses with the newly-active model name, or None if cancelled."""

    CSS = _PICKER_CSS
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._group = model_registry.get_active_group()

    def compose(self) -> ComposeResult:
        if self._group is None:
            with Vertical(id="picker-box"):
                yield Static("No login configured yet — run /login first.", classes="picker-err")
            return
        active_model = self._group.get("active_model")
        options = [
            (m + (" (current)" if m == active_model else ""), m)
            for m in self._group.get("models", [])
        ]
        options.append(("+ Add new model...", "__add__"))
        with Vertical(id="picker-box"):
            yield Static(f"Models for login {self._group['name']!r} — enter to select, esc to cancel")
            yield Select(options, id="picker-select", allow_blank=True)
            with Vertical(id="add-form"):
                yield Input(placeholder="model (e.g. claude-opus-5)", id="f-model")
                yield Button("Save", id="f-save", variant="primary")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "picker-select":
            return
        if event.value == "__add__":
            self.query_one("#add-form").add_class("visible")
            self.query_one("#f-model", Input).focus()
        elif event.value is not Select.BLANK:
            model_registry.set_active_model(self._group["name"], event.value)
            self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "f-save":
            return
        model_name = self.query_one("#f-model", Input).value.strip()
        box = self.query_one("#picker-box")
        for old in box.query(".picker-err"):
            old.remove()
        if not model_name:
            box.mount(Static("model name is required", classes="picker-err"))
            return
        model_registry.add_model(self._group["name"], model_name)
        self.dismiss(model_name)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ----------------------------------------------------------------------
# PromptInput — multi-line prompt box. Enter submits, shift+enter inserts
# a newline; TextArea gives soft-wrap and multi-line height for free.
# ----------------------------------------------------------------------

class PromptInput(TextArea):
    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    BINDINGS = [Binding("shift+enter", "insert_newline", "New line")]

    def __init__(self, **kwargs) -> None:
        super().__init__(show_line_numbers=False, soft_wrap=True, tab_behavior="focus", **kwargs)

    def action_insert_newline(self) -> None:
        self.insert("\n")

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            text = self.text
            self.text = ""
            self.post_message(self.Submitted(text))
            return
        await super()._on_key(event)


# ----------------------------------------------------------------------
# TuiRenderer — Textual App + Renderer protocol
# ----------------------------------------------------------------------

class TuiRenderer(App):
    """Textual app. Renderer-protocol methods run on a worker thread and
    marshal onto the UI thread via call_from_thread."""

    CSS = """
    Screen { padding: 0 0 1 0; }
    #status { background: $primary-darken-1; color: $text; height: 1; margin-bottom: 1; }
    #transcript { padding: 0 1 1 1; }
    .step { text-style: bold; }
    .tool { color: $text-muted; }
    .ok { color: $success; }
    .err { color: $error; text-style: bold; }
    .head { text-style: bold; }
    .dim { color: $text-muted; }
    .add { background: $success-muted; color: $text; }
    .del { background: $error-muted; color: $text; }
    .result { text-style: bold; }
    .usermsg { border: round $accent; padding: 0 1; margin: 0 0 1 0; }
    Collapsible { padding: 0; margin: 0 0 1 0; }
    Collapsible.err > CollapsibleTitle { color: $error; }
    Collapsible.ok > CollapsibleTitle { color: $success; }
    #hints { color: $text-muted; padding: 0 1; height: auto; display: none; margin-bottom: 1; }
    #hints.visible { display: block; }
    #input { height: auto; max-height: 10; margin-bottom: 1; }
    #footer-session { color: $text-muted; padding: 0 1; height: 1; }
    #footer-model { color: $text-muted; padding: 0 1; height: 1; }
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
        self.session_name = ""
        active_group = model_registry.get_active_group()
        self.group_name = active_group["name"] if active_group else ""
        self.step_count = 0
        self.step_total = 0
        self._input_queue: "queue.Queue[str | None]" = queue.Queue()
        self._batch: list[dict] | None = None
        self._read_widget: Static | None = None
        self._read_hints: list[str] = []
        # Cross-step run: consecutive steps that are each a single read_file
        # (or bash) batch collapse into one growing Collapsible instead of
        # one per step — a paginated multi-call read shouldn't cost N rows.
        self._run_kind: str | None = None
        self._run_collapsible: Collapsible | None = None
        self._run_hints: list[str] = []
        self._run_has_error = False
        self._transcript: VerticalScroll | None = None
        self._status: Static | None = None
        self._input: "PromptInput | None" = None
        self._hints: Static | None = None
        self._footer_session: Static | None = None
        self._footer_model: Static | None = None
        # Set once the app's event loop is actually running — call_from_thread
        # raises if invoked before this, so the session worker thread must
        # wait on it before touching any Renderer method.
        self._loop_ready = threading.Event()
        # Set by /clear (UI thread), consumed by agent.py's session loop
        # (worker thread) — the renderer doesn't own `messages`, so it can
        # only ask the loop to reset it, not do so itself.
        self._clear_requested = threading.Event()

    def wait_until_ready(self, timeout: float | None = None) -> None:
        self._loop_ready.wait(timeout)

    def take_clear_request(self) -> bool:
        """True (once) if /clear was run since the last call."""
        was_set = self._clear_requested.is_set()
        self._clear_requested.clear()
        return was_set

    # ---- Textual app ----

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        yield VerticalScroll(id="transcript")
        yield Static(id="hints")
        yield PromptInput(id="input")
        yield Static(id="footer-session")
        yield Static(id="footer-model")

    def on_mount(self) -> None:
        self._transcript = self.query_one("#transcript", VerticalScroll)
        self._status = self.query_one("#status", Static)
        self._input = self.query_one("#input", PromptInput)
        self._hints = self.query_one("#hints", Static)
        self._footer_session = self.query_one("#footer-session", Static)
        self._footer_model = self.query_one("#footer-model", Static)
        self._refresh_status()
        self._mount_line(
            "Tip: type / to see available commands (e.g. /help). Enter to send, shift+enter for a new line.",
            "dim",
        )
        self._input.focus()
        self._loop_ready.set()

    def on_prompt_input_submitted(self, event: "PromptInput.Submitted") -> None:
        self._hints.set_class(False, "visible")
        self._input_queue.put(event.text)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        if text.startswith("/") and " " not in text and "\n" not in text:
            prefix = text[1:].lower()
            matches = [(n, d) for n, d in SLASH_COMMANDS.items() if n.startswith(prefix)]
            if matches:
                self._hints.update(Text("\n".join(f"/{n} — {d}" for n, d in matches)))
                self._hints.set_class(True, "visible")
                return
        self._hints.set_class(False, "visible")

    def action_quit_app(self) -> None:
        self._input_queue.put(None)
        self.exit()

    def _refresh_status(self) -> None:
        if self._status is not None:
            self._status.update(Text(format_status(self.step_count, self.step_total)))
        if self._footer_session is not None and self._footer_model is not None:
            session_line, model_line = format_footer(
                self.session_name, self.session_id, self.group_name, self.model_name
            )
            self._footer_session.update(Text(session_line))
            self._footer_model.update(Text(model_line))

    def _mount_line(self, text: str, cls: str = "") -> Static:
        w = Static(Text(text), classes=cls)
        self._transcript.mount(w)
        self._transcript.scroll_end(animate=False)
        return w

    def _update_line(self, widget: Static, text: str) -> None:
        widget.update(Text(text))
        self._transcript.scroll_end(animate=False)

    def _mount_user_message(self, text: str) -> None:
        self._transcript.mount(Static(Text(text), classes="usermsg"))
        self._transcript.scroll_end(animate=False)

    def _clear_transcript(self) -> None:
        for child in list(self._transcript.children):
            child.remove()
        self._run_kind = None
        self._run_collapsible = None
        self._mount_line("[cleared] new conversation started", "dim")

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
        self._read_widget = None
        self._read_hints = []
        self.call_from_thread(self._refresh_status)

    def tool_call(self, num: int, name: str, args_json: str) -> None:
        hint = _arg_hint(name, args_json)
        if name == "read_file" and self._read_widget is not None:
            # Collapse consecutive reads into one running line instead of
            # a new one per call — a multi-file read shouldn't cost N lines.
            self._read_hints.append(hint or "?")
            call_text = "  tool: read " + ", ".join(self._read_hints)
            self.call_from_thread(self._update_line, self._read_widget, call_text)
            if self._batch is not None:
                self._batch.append({
                    "name": name, "hint": hint, "ok": True, "_open": True,
                    "widgets": [], "details": [("tool", call_text)],
                })
            return
        truncated = (args_json or "")[:200].replace("\n", " ")
        call_text = f"  tool: {name} {truncated}"
        widget = self.call_from_thread(self._mount_line, call_text, "tool")
        self._read_widget = widget if name == "read_file" else None
        self._read_hints = [hint or "?"] if name == "read_file" else []
        if self._batch is not None:
            self._batch.append({
                "name": name,
                "hint": hint,
                "ok": True,
                "_open": True,
                "widgets": [widget],
                "details": [("tool", call_text)],
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
        if tool_name == "read_file" and not is_error:
            return  # the running "read x, y" line already covers this
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

    def _extend_run(self, others: list[dict], kind: str) -> None:
        collapsible = self._run_collapsible
        contents = collapsible.query_one(Collapsible.Contents)
        for e in others:
            self._run_hints.append(e["hint"] or "?")
            if not e["ok"]:
                self._run_has_error = True
            for dkind, dtext in e["details"]:
                contents.mount(Static(Text(dtext), classes=dkind))
        collapsible.title = run_title(kind, self._run_hints, self._run_has_error)
        collapsible.set_class(self._run_has_error, "err")
        collapsible.set_class(not self._run_has_error, "ok")

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
            if not others:
                return
            names = {e["name"] for e in others}
            kind = next(iter(names)) if len(names) == 1 and next(iter(names)) in RUNNABLE_KINDS else None

            if kind is not None and kind == self._run_kind and self._run_collapsible is not None:
                self._extend_run(others, kind)
                self._transcript.scroll_end(animate=False)
                return

            text, has_error = summarize_batch([(e["name"], e["hint"], e["ok"]) for e in others])
            detail_widgets = []
            for e in others:
                for dkind, dtext in e["details"]:
                    detail_widgets.append(Static(Text(dtext), classes=dkind))
            title = run_title(kind, [e["hint"] or "?" for e in others], has_error) if kind else f"{'✖' if has_error else '✔'} {text}"
            collapsible = Collapsible(
                *detail_widgets,
                title=title,
                collapsed=True,
                classes="err" if has_error else "ok",
            )
            self._transcript.mount(collapsible)
            self._transcript.scroll_end(animate=False)
            if kind is not None:
                self._run_kind = kind
                self._run_collapsible = collapsible
                self._run_hints = [e["hint"] or "?" for e in others]
                self._run_has_error = has_error
            else:
                self._run_kind = None
                self._run_collapsible = None

        self.call_from_thread(finish)

    def step(self, num: int, total: int, text: str) -> None:
        self.step_count = num
        self.step_total = total
        stripped = (text or "").strip()
        if stripped:
            self.call_from_thread(self._mount_line, stripped)
        self.call_from_thread(self._refresh_status)

    def final(self, summary: str) -> None:
        self.call_from_thread(self._mount_line, f"result: {summary}", "result")

    def cancelled(self) -> None:
        self.call_from_thread(self._mount_line, "[cancelled]", "err")

    # ---- slash commands (handled locally, never sent to the agent) ----

    def _on_model_picked(self, name: str | None) -> None:
        if name is None:
            return
        self.model_name = name
        self._refresh_status()
        self._mount_line(f"model set to {name!r}", "dim")

    def _on_login_picked(self, name: str | None) -> None:
        if name is None:
            return
        group = model_registry.get_active_group() or {}
        self.group_name = name
        self.model_name = group.get("active_model") or self.model_name
        self._refresh_status()
        self._mount_line(f"login set to {name!r} ({group.get('provider')}/{self.model_name})", "dim")

    def _handle_slash_command(self, line: str) -> bool:
        """Run a /command mounted on the UI thread. Returns True if the app
        should exit (read_line() should stop and return None)."""
        cmd, _, rest = line[1:].partition(" ")
        cmd = cmd.strip().lower()
        rest = rest.strip()
        if cmd in ("quit", "exit"):
            self.exit()
            return True
        if cmd in ("name", "rename"):
            if not rest:
                self._mount_line("usage: /name <text>", "err")
            else:
                self.session_name = rest
                self._refresh_status()
                self._mount_line(f"session renamed to {rest!r}", "dim")
        elif cmd == "clear":
            self._clear_transcript()
            self._clear_requested.set()
        elif cmd == "login":
            self.push_screen(LoginPickerScreen(), self._on_login_picked)
        elif cmd == "model":
            self.push_screen(ModelPickerScreen(), self._on_model_picked)
        elif cmd == "help":
            self._mount_line(
                "\n".join(f"/{n} — {d}" for n, d in SLASH_COMMANDS.items()), "dim"
            )
        else:
            self._mount_line(f"unknown command: /{cmd} (try /help)", "err")
        return False

    # ---- input (called from the session worker thread) ----

    def read_line(self, prompt: str = "> ") -> str | None:
        while True:
            self.call_from_thread(self._set_input_enabled, True)
            line = self._input_queue.get()
            self.call_from_thread(self._set_input_enabled, False)
            if line is None:
                return None
            if line.startswith("/"):
                should_exit = self.call_from_thread(self._handle_slash_command, line)
                if should_exit:
                    return None
                continue
            self.call_from_thread(self._mount_user_message, line)
            return line
