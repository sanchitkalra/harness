"""Tools: everything that touches the workspace. No printing, no model calls.

Sandbox rule lives here: every path goes through resolve(), which
rejects anything escaping the workspace root.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

MAX_OUTPUT_CHARS = 4000
DEFAULT_TIMEOUT_S = 30
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
