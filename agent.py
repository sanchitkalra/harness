"""Mini coding-agent harness: LLM + tool loop you can read in one sitting.

Run:
  export MODEL_API_KEY=...             # Muse Spark key from https://dev.meta.ai
  # optional: export MUSE_SPARK_MODEL=muse-spark-1.1 MUSE_SPARK_BASE_URL=...
  # (or use OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL instead)
  source .venv/bin/activate
  python -m agent "list files in this workspace" --workspace .
  python -m agent --smoke --workspace .  # no API key needed
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import ui
from model import ApiError, NetworkError, llm_call, llm_config, resolve_provider
from tools import (  # re-exported: tests and callers keep working via agent.*
    MAX_OUTPUT_CHARS,
    TOOLS,
    dispatch,
    load_skills,
    tool_bash,
    tool_edit,
    tool_read,
    tool_read_skill,
    tool_search,
    tool_write,
    truncate,
)

DEFAULT_MAX_STEPS = 30
MAX_REPEAT_CALLS = 3  # same tool+args this many times -> stop: no_progress
MAX_IDLE_TURNS = 3  # model replies with no tool call this many times in a row -> stop


SYSTEM = (
    "You are a minimal coding agent. Work inside the workspace only. "
    "Use tools to read before editing, then verify with bash (e.g. pytest -q). "
    "Large outputs are paged by lines; re-read with offset to continue. "
    "Call done with a short summary when finished. "
    "When the task is verified, call done immediately — do not keep narrating. "
    "If a tool returns an error, fix your approach instead of repeating it. "
    "You may update MEMORY.md via edit_file/write_file to persist durable facts across sessions."
)


def load_instructions(root: Path) -> str:
    for name in ("AGENTS.md", "CLAUDE.md"):
        p = root / name
        try:
            if p.is_file():
                text = p.read_text(encoding="utf-8")
                return text[:2000]
        except Exception:
            continue
    return ""


def load_memory(root: Path) -> str:
    p = root / "MEMORY.md"
    try:
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            return text[:2000]
    except Exception:
        pass
    return ""


def new_conversation(task: str, root: Path) -> list[dict]:
    """Build initial messages list with system prompt and optional user task."""
    instr = load_instructions(root)
    mem = load_memory(root)
    sys_content = SYSTEM + (f"\n\nWorkspace instructions:\n{instr}" if instr else "")
    if mem:
        sys_content += f"\n\nLong-term memory:\n{mem}"
    try:
        skills = load_skills(root)
        if skills:
            lines = []
            for name in sorted(skills.keys()):
                desc = skills[name].get("description", "")
                if desc:
                    lines.append(f"- {name}: {desc}")
                else:
                    lines.append(f"- {name}")
            skills_block = "\n".join(lines)
            sys_content += f"\n\nAvailable skills (use read_skill to load full instructions):\n{skills_block}"
    except Exception:
        pass
    messages: list[dict] = [
        {"role": "system", "content": sys_content},
    ]
    if task:
        messages.append({"role": "user", "content": task})
    return messages


def drive(
    messages: list[dict],
    root: Path,
    max_steps: int = DEFAULT_MAX_STEPS,
    log_path: Path | None = None,
    parent_id: str | None = None,
    renderer: ui.Renderer | None = None,
) -> str:
    """Run the agent step loop over an existing messages list.
    Mutates messages by appending assistant and tool messages.
    If log_path is given, writes a session header (task = last user message)
    and logs each step.
    """
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            model_name = resolve_provider().model
        except Exception:
            model_name = "unknown"
        # Derive task for header from last user message, else empty
        task_for_header = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                task_for_header = m.get("content", "") or ""
                break
        header = {
            "type": "session",
            "id": log_path.stem,
            "ts": datetime.now(timezone.utc).isoformat(),
            "task": task_for_header,
            "parent_id": parent_id,
            "workspace": str(root),
            "model": model_name,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")

    if renderer is None:
        renderer = ui.PrintRenderer()

    seen: dict[tuple, int] = {}  # (tool, canonical args) -> times seen; bounded by repeat limit
    idle_turns = 0
    for step in range(1, max_steps + 1):
        try:
            msg = llm_call(messages, TOOLS)
        except ApiError as e:
            return _stop(log_path, step, f"stopped: api ({e}, {step}/{max_steps} steps)")
        except NetworkError as e:  # network down even after retries: stop, don't crash
            return _stop(log_path, step, f"stopped: network ({e}, {step}/{max_steps} steps)")
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        text = (msg.get("content") or "").strip()
        if text:
            renderer.step(step, max_steps, text)
        if not calls:  # model talked without acting: nudge it back to tools
            idle_turns += 1
            if idle_turns >= MAX_IDLE_TURNS:
                return _stop(log_path, step, f"stopped: idle (no tool call for {idle_turns} turns, {step}/{max_steps} steps)")
            messages.append({"role": "user", "content": "Continue: call a tool or done."})
            continue
        idle_turns = 0
        renderer.begin_tools(step)
        try:
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    messages.append({"role": "tool", "tool_call_id": c["id"],
                                     "content": f"error: bad_args: arguments are not valid JSON ({e}); retry with quoted strings"})
                    continue
                key = (name, json.dumps(args, sort_keys=True))
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 1:  # repeat: warn, and stop if it keeps going nowhere
                    if seen[key] >= MAX_REPEAT_CALLS:
                        return _stop(log_path, step, f"stopped: no_progress (repeated {name} {MAX_REPEAT_CALLS}x, {step}/{max_steps} steps)")
                    messages.append({"role": "tool", "tool_call_id": c["id"],
                                     "content": f"error: repeated_call: identical call to {name} ({seen[key]}x); try something different"})
                    continue
                renderer.tool_call(step, name, json.dumps(args))
                if name == "done":
                    final = args.get("summary", "")
                    log(log_path, {"step": step, "tool": name, "args": args, "result": final})
                    renderer.end_tools()
                    return final
                try:
                    result = dispatch(root, name, args)
                except Exception as e:  # surface sandbox errors to the model, don't crash
                    result = f"error: {e}"
                renderer.tool_result(name, result)
                log(log_path, {"step": step, "tool": name, "args": args, "result": result[:2000]})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        finally:
            renderer.end_tools()
    return _stop(log_path, max_steps, f"stopped: max_steps ({max_steps} steps without done)")


def _stop(log_path: Path | None, step: int, msg: str) -> str:
    """Log a terminal outcome so failures leave a trace, then return it."""
    log(log_path, {"step": step, "tool": "stopped", "args": {}, "result": msg})
    return msg


def run(task: str, root: Path, max_steps: int = DEFAULT_MAX_STEPS, log_path: Path | None = None, parent_id: str | None = None) -> str:
    messages = new_conversation(task, root)
    return drive(messages, root, max_steps, log_path, parent_id)


def log(path: Path | None, entry: dict) -> None:
    if path is None:
        return
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def smoke_test(root: Path) -> None:
    """No-API check: exercises sandbox, edit uniqueness, truncation, bash."""
    demo = root / "smoke_demo.txt"
    demo.write_text("hello\n", encoding="utf-8")
    assert "hello" in tool_read(root, "smoke_demo.txt")
    try:
        tool_read(root, "../outside.txt")
        raise AssertionError("path traversal should have raised")
    except ValueError:
        pass
    assert "not found" in tool_edit(root, "smoke_demo.txt", "zzz", "y")
    demo.write_text("a a a", encoding="utf-8")
    assert "matches 3 times" in tool_edit(root, "smoke_demo.txt", "a", "b")
    assert "truncated" in truncate("x" * (MAX_OUTPUT_CHARS + 10))
    r = tool_bash(root, f"{sys.executable} -c \"print('hi')\"")
    assert "exit=0" in r and "hi" in r
    demo.unlink(missing_ok=True)
    print("smoke ok: sandbox, edit guards, truncation, bash all pass")


def sessions_dir(root: Path) -> Path:
    d = root / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_sessions(root: Path) -> None:
    """List sessions/*.jsonl files: id, ts, task from header, or (no header)."""
    sdir = sessions_dir(root)
    for fp in sorted(sdir.glob("*.jsonl"), key=lambda p: p.name):
        try:
            line = fp.read_text(encoding="utf-8").splitlines()[0].strip()
            if not line:
                raise ValueError("empty")
            obj = json.loads(line)
            if obj.get("type") != "session":
                raise ValueError("not session")
            sid = obj.get("id") or fp.stem
            ts = obj.get("ts", "")
            task = " ".join(str(obj.get("task", "")).splitlines()).strip()
            if task:
                print(f"{sid} {ts} {task}")
            else:
                print(f"{sid} {ts}".rstrip())
        except Exception:
            print(f"{fp.stem} (no header)")


def _unique_log_path(root: Path) -> Path:
    """Generate a unique log path using UTC timestamp with microseconds."""
    sdir = sessions_dir(root)
    for _ in range(10):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        p = sdir / f"{stamp}.jsonl"
        if not p.exists():
            return p
        time.sleep(0.001)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    p = sdir / f"{stamp}.jsonl"
    counter = 0
    while p.exists():
        counter += 1
        p = sdir / f"{stamp}-{counter}.jsonl"
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Mini coding-agent harness")
    ap.add_argument("--version", action="version", version="harness 0.1.0")
    ap.add_argument("task", nargs="?", default="", help="task text")
    ap.add_argument("--workspace", default=".", help="workspace root")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--smoke", action="store_true", help="run no-API smoke test")
    ap.add_argument("--list", dest="list_flag", action="store_true", help="list sessions/*.jsonl (id, ts, task) and exit")
    ap.add_argument("--fork", dest="fork_id", default=None, help="fork from existing session id (root/sessions/<id>.jsonl)")
    ap.add_argument("-i", "--interactive", dest="interactive", action="store_true", help="interactive REPL mode")
    ap.add_argument("--no-tui", dest="no_tui", action="store_true", help="disable Textual TUI in interactive mode")
    args = ap.parse_args()
    root = Path(args.workspace).resolve()
    if args.list_flag:
        list_sessions(root)
        return
    if args.smoke:
        smoke_test(root)
        return
    if not args.task and not args.interactive:
        ap.error("give a TASK or pass --smoke")
    parent_id_for_header: str | None = None
    task_text = args.task
    if args.fork_id is not None:
        parent_file = sessions_dir(root) / f"{args.fork_id}.jsonl"
        if not parent_file.is_file():
            ap.error(f"fork: parent session file not found: {parent_file} (id {args.fork_id!r})")
        try:
            first_line = parent_file.read_text(encoding="utf-8").splitlines()[0].strip()
            parent_header = json.loads(first_line)
            if parent_header.get("type") != "session":
                raise ValueError("header type is not 'session'")
        except Exception as e:
            ap.error(f"fork: parent session header missing or invalid in {parent_file}: {e}")
        parent_id_for_header = parent_header.get("id") or args.fork_id
        parent_task = parent_header.get("task", "")
        try:
            all_lines = [ln for ln in parent_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            step_count = max(0, len(all_lines) - 1)
        except Exception:
            step_count = 0
        context_line = f"Forked from session {parent_id_for_header}: task={parent_task!r} steps={step_count}"
        task_text = context_line + "\n" + task_text
    try:
        resolve_provider()
    except RuntimeError as e:
        ap.error(f"{e} (or use --smoke for the no-API check)")

    if args.interactive:
        use_tui = False
        if not args.no_tui:
            try:
                use_tui = sys.stdout.isatty()
            except Exception:
                use_tui = False

        renderer = None
        if use_tui:
            try:
                import tui as tui_mod  # local
                try:
                    model_name = resolve_provider().model
                except Exception:
                    model_name = "unknown"
                renderer = tui_mod.TuiRenderer(model_name=model_name, session_id=parent_id_for_header or "")
            except Exception:
                use_tui = False
                renderer = None
        if renderer is None:
            renderer = ui.PrintRenderer()

        def get_input_line(prompt: str = "> ") -> str | None:
            if use_tui:
                return renderer.read_line(prompt)
            try:
                return input(prompt)
            except EOFError:
                return None
            # KeyboardInterrupt propagates to the outer handler (continue).

        def show_final(r: str | None) -> None:
            if r is None:
                return
            renderer.final(r)

        def run_drive_with_renderer(msgs: list[dict], lpath: Path, pid: str | None) -> str | None:
            try:
                return drive(msgs, root, args.max_steps, lpath, parent_id=pid, renderer=renderer)
            except KeyboardInterrupt:
                if use_tui:
                    renderer.cancelled()
                else:
                    print("\n[cancelled]")
                return None

        def run_interactive_session() -> None:
            messages = new_conversation(task_text, root)
            prev_id = parent_id_for_header
            if task_text:
                log_path = _unique_log_path(root)
                if use_tui:
                    renderer.update_status(session_id=log_path.stem)
                result = run_drive_with_renderer(messages, log_path, prev_id)
                if result is not None:
                    show_final(result)
                    prev_id = log_path.stem
                    if use_tui:
                        renderer.update_status(session_id=prev_id)

            while True:
                try:
                    line = get_input_line("> ")
                except KeyboardInterrupt:
                    if not use_tui:
                        print()
                    continue
                if line is None:
                    if not use_tui:
                        print()
                    break
                if not line.strip():
                    if use_tui:
                        # Enter sends; empty input is a no-op, exit via Ctrl-D/Ctrl-C.
                        continue
                    break
                checkpoint = len(messages)
                messages.append({"role": "user", "content": line})
                log_path = _unique_log_path(root)
                if use_tui:
                    renderer.update_status(session_id=log_path.stem)
                result = run_drive_with_renderer(messages, log_path, prev_id)
                if result is None:
                    # cancelled via KeyboardInterrupt inside drive — pop message
                    del messages[checkpoint:]
                    continue
                show_final(result)
                prev_id = log_path.stem
                if use_tui:
                    renderer.update_status(session_id=prev_id)

        if use_tui:
            def worker() -> None:
                # Wait for the app's event loop to actually be running —
                # call_from_thread (used by every Renderer method) raises
                # if invoked any earlier.
                renderer.wait_until_ready(timeout=10)
                try:
                    run_interactive_session()
                finally:
                    try:
                        renderer.call_from_thread(renderer.exit)
                    except Exception:
                        pass  # app may already be exiting (e.g. Ctrl-C from the UI thread)

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            renderer.run()  # blocks main thread until renderer.exit()
            t.join(timeout=2)
        else:
            run_interactive_session()
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    result = run(task_text, root, args.max_steps, sessions_dir(root) / f"{stamp}.jsonl", parent_id=parent_id_for_header)
    ui.final(result)


if __name__ == "__main__":
    main()
