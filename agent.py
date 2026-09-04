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
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MAX_OUTPUT_CHARS = 4000
DEFAULT_MAX_STEPS = 30
DEFAULT_TIMEOUT_S = 30
MAX_REPEAT_CALLS = 3  # same tool+args this many times -> stop: no_progress
MAX_IDLE_TURNS = 3  # model replies with no tool call this many times in a row -> stop
BLOCKED_BASH_PATTERNS = ["rm -rf /", "rm -rf ~", ":(){", "mkfs", "dd of=/dev"]

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file inside the workspace. Output is paged by lines (default 200); pass offset/limit to read the rest.",
        "parameters": {"path": "workspace-relative path, e.g. agent.py", "offset": "optional first line, 0-based (default 0)", "limit": "optional max lines (default 200)"},
    },
    {
        "name": "edit_file",
        "description": "Replace one unique exact string in a file.",
        "parameters": {"path": "file to edit", "find": "exact text", "replace": "replacement"},
    },
    {
        "name": "write_file",
        "description": "Create a file inside the workspace. Refuses to overwrite unless overwrite is 'true'. Parent dir must already exist.",
        "parameters": {"path": "file to create", "content": "full file text", "overwrite": "optional 'true' to overwrite"},
    },
    {
        "name": "bash",
        "description": "Run a shell command with cwd=workspace. Returns exit code, stdout, stderr.",
        "parameters": {"command": "e.g. pytest -q", "timeout_s": "optional seconds (default 30)"},
    },
    {
        "name": "done",
        "description": "Call when the task is complete. Summarise what changed.",
        "parameters": {"summary": "short result for the user"},
    },
]


def resolve(root: Path, rel: str) -> Path:
    """Sandbox: reject anything that escapes the workspace root."""
    p = (root / rel).resolve()
    if p != root.resolve() and root.resolve() not in p.parents:
        raise ValueError(f"blocked: {rel!r} escapes workspace {root}")
    return p


def truncate(s: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated {len(s) - limit} chars]"


def tool_read(root: Path, path: str, offset: int = 0, limit: int = 200) -> str:
    try:
        off = max(0, int(offset or 0))
    except (TypeError, ValueError):
        return "error: bad_args: 'offset' must be an integer, e.g. {'path': 'agent.py', 'offset': 200}"
    try:
        lim = max(1, int(limit or 200))
    except (TypeError, ValueError):
        return "error: bad_args: 'limit' must be an integer, e.g. {'path': 'agent.py', 'limit': 200}"
    p = resolve(root, path)
    lines = p.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    page = lines[off:off + lim]
    if page:
        shown = f"lines {off + 1}-{off + len(page)} of {total}"
    else:
        shown = f"lines {off + 1}-{off} of {total} (past end)"
    body = "\n".join(page)
    if len(body) > MAX_OUTPUT_CHARS:  # very long lines: char-cap the page, keep it recoverable
        body = body[:MAX_OUTPUT_CHARS] + f"\n... [truncated {len(body) - MAX_OUTPUT_CHARS} chars; re-read with a smaller limit]"
    out = f"{shown}\n{body}"
    if off + lim < total:
        out += f"\n... [more: re-read {path!r} with offset={off + lim} to continue]"
    return out


def tool_edit(root: Path, path: str, find: str, replace: str) -> str:
    p = resolve(root, path)
    text = p.read_text(encoding="utf-8")
    n = text.count(find)
    if n == 0:
        return "error: 'find' string not found (0 matches)"
    if n > 1:
        return f"error: 'find' matches {n} times; include more context to make it unique"
    p.write_text(text.replace(find, replace), encoding="utf-8")
    return f"ok: edited {path}"


def tool_write(root: Path, path: str, content: str, overwrite: str = "") -> str:
    if not path:
        return "error: missing_arg: 'path' is required, e.g. {'path': 'notes.txt', 'content': '...'}"
    if content is None:
        return "error: missing_arg: 'content' is required, e.g. {'path': 'notes.txt', 'content': '...'}"
    p = resolve(root, path)
    want_overwrite = str(overwrite).lower() in ("1", "true", "yes")
    if p.is_dir():
        return f"error: is_dir: {path!r} is a directory, pick a file path instead"
    if p.exists() and not want_overwrite:
        return f"error: exists: {path!r} already exists; pass 'overwrite': 'true' to replace it"
    if not p.parent.exists():
        return f"error: no_parent: parent dir {str(p.parent.relative_to(root.resolve()))!r} does not exist; run mkdir via bash first"
    p.write_text(content, encoding="utf-8")
    action = "overwrote" if want_overwrite and p.exists() else "wrote"
    return f"ok: {action} {path} ({len(content)} chars)"


def tool_bash(root: Path, command: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> str:
    if not (command or "").strip():
        return "error: missing_arg: 'command' is required, e.g. {'command': 'pytest -q'}"
    for pat in BLOCKED_BASH_PATTERNS:
        if pat in command:
            return f"error: blocked: command contains {pat!r}; try something narrower"
    try:
        proc = subprocess.run(
            command, shell=True, cwd=root, capture_output=True,
            text=True, timeout=float(timeout_s or DEFAULT_TIMEOUT_S),
        )
    except subprocess.TimeoutExpired:
        return f"error: timeout: timed out after {timeout_s}s (cwd={root})"
    except OSError as e:
        return f"error: exec: could not run command (cwd={root}): {e}"
    out = f"exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return truncate(out)


SYSTEM = (
    "You are a minimal coding agent. Work inside the workspace only. "
    "Use tools to read before editing, then verify with bash (e.g. pytest -q). "
    "Large outputs are paged by lines; re-read with offset to continue. "
    "Call done with a short summary when finished. "
    "When the task is verified, call done immediately — do not keep narrating. "
    "If a tool returns an error, fix your approach instead of repeating it."
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


META_BASE_URL = "https://api.meta.ai/v1"
META_DEFAULT_MODEL = "muse-spark-1.1"


def llm_config() -> tuple[str, str, str]:
    """Resolve (base_url, model, key) from env. OpenAI vars win when set."""
    if "OPENAI_API_KEY" in os.environ:
        return (
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            os.environ["OPENAI_API_KEY"],
        )
    for var in ("MODEL_API_KEY", "MUSE_SPARK_API_KEY", "META_API_KEY"):
        if os.environ.get(var):
            return (
                os.environ.get("MUSE_SPARK_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL", META_BASE_URL),
                os.environ.get("MUSE_SPARK_MODEL")
                or os.environ.get("OPENAI_MODEL", META_DEFAULT_MODEL),
                os.environ[var],
            )
    raise RuntimeError("set MODEL_API_KEY (Muse Spark) or OPENAI_API_KEY")


def llm_call(messages: list[dict], tools: list[dict]) -> dict:
    """One chat-completions call (OpenAI-compatible). Returns parsed message."""
    base, model, key = llm_config()
    schema = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in t["parameters"]},
                },
            },
        }
        for t in tools
    ]
    body = json.dumps({"model": model, "messages": messages, "tools": schema}).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return data["choices"][0]["message"]


def dispatch(root: Path, name: str, args: dict) -> str:
    if name == "read_file":
        if not args.get("path"):
            return "error: missing_arg: 'path' is required, e.g. {'path': 'agent.py'}"
        return tool_read(root, args.get("path", ""), args.get("offset", 0), args.get("limit", 200))
    if name == "edit_file":
        for k in ("path", "find", "replace"):
            if k not in args:
                return f"error: missing_arg: {k!r} is required, e.g. {{'path': 'f.txt', 'find': 'a', 'replace': 'b'}}"
        return tool_edit(root, args.get("path", ""), args.get("find", ""), args.get("replace", ""))
    if name == "write_file":
        if "path" not in args or "content" not in args:
            return "error: missing_arg: 'path' and 'content' are required, e.g. {'path': 'notes.txt', 'content': '...'}"
        return tool_write(root, args.get("path", ""), args.get("content"), args.get("overwrite", ""))
    if name == "bash":
        if "command" not in args:
            return "error: missing_arg: 'command' is required, e.g. {'command': 'pytest -q'}"
        return tool_bash(root, args.get("command", ""), float(args.get("timeout_s") or DEFAULT_TIMEOUT_S))
    if name == "done":
        return "done"
    valid = ", ".join(t["name"] for t in TOOLS)
    return f"error: unknown_tool: {name!r} is not a tool; valid tools are: {valid}"


def run(task: str, root: Path, max_steps: int = DEFAULT_MAX_STEPS, log_path: Path | None = None, parent_id: str | None = None) -> str:
    instr = load_instructions(root)
    sys_content = SYSTEM + (f"\n\nWorkspace instructions:\n{instr}" if instr else "")
    messages: list[dict] = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": task},
    ]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _base, _model, _key = llm_config()
            model_name = _model
        except Exception:
            model_name = "unknown"
        header = {
            "type": "session",
            "id": log_path.stem,
            "ts": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "parent_id": parent_id,
            "workspace": str(root),
            "model": model_name,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
    seen: dict[tuple, int] = {}  # (tool, canonical args) -> times seen; bounded by repeat limit
    idle_turns = 0
    for step in range(1, max_steps + 1):
        msg = llm_call(messages, TOOLS)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        text = (msg.get("content") or "").strip()
        if text:
            print(f"[step {step}] {text[:300]}")
        if not calls:  # model talked without acting: nudge it back to tools
            idle_turns += 1
            if idle_turns >= MAX_IDLE_TURNS:
                return f"stopped: idle (no tool call for {idle_turns} turns, {step}/{max_steps} steps)"
            messages.append({"role": "user", "content": "Continue: call a tool or done."})
            continue
        idle_turns = 0
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
                    return f"stopped: no_progress (repeated {name} {MAX_REPEAT_CALLS}x, {step}/{max_steps} steps)"
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": f"error: repeated_call: identical call to {name} ({seen[key]}x); try something different"})
                continue
            print(f"[step {step}] tool: {name} {json.dumps(args)[:200]}")
            if name == "done":
                final = args.get("summary", "")
                log(log_path, {"step": step, "tool": name, "args": args, "result": final})
                return final
            try:
                result = dispatch(root, name, args)
            except Exception as e:  # surface sandbox errors to the model, don't crash
                result = f"error: {e}"
            print(f"  -> {result[:300]}")
            log(log_path, {"step": step, "tool": name, "args": args, "result": result[:2000]})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    return f"stopped: max_steps ({max_steps} steps without done)"


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Mini coding-agent harness")
    ap.add_argument("task", nargs="?", default="", help="task text")
    ap.add_argument("--workspace", default=".", help="workspace root")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--smoke", action="store_true", help="run no-API smoke test")
    ap.add_argument("--list", dest="list_flag", action="store_true", help="list sessions/*.jsonl (id, ts, task) and exit")
    ap.add_argument("--fork", dest="fork_id", default=None, help="fork from existing session id (root/sessions/<id>.jsonl)")
    args = ap.parse_args()
    root = Path(args.workspace).resolve()
    if args.list_flag:
        list_sessions(root)
        return
    if args.smoke:
        smoke_test(root)
        return
    if not args.task:
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
        llm_config()
    except RuntimeError as e:
        ap.error(f"{e} (or use --smoke for the no-API check)")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    result = run(task_text, root, args.max_steps, sessions_dir(root) / f"{stamp}.jsonl", parent_id=parent_id_for_header)
    print(f"\nresult: {result}")


if __name__ == "__main__":
    main()
