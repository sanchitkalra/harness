"""Tools: everything that touches the workspace. No printing, no model calls.

Sandbox rule lives here: every path goes through resolve(), which
rejects anything escaping the workspace root.
"""
from __future__ import annotations

import difflib
import html
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MAX_OUTPUT_CHARS = 4000
DEFAULT_TIMEOUT_S = 30
# Web-search retry budget (seconds of total backoff sleep), mirroring model.py.
# Kept local: tools never imports model internals.
RETRY_BUDGET_S = 30.0
RETRY_BASE_S = 1.0
RETRY_CAP_S = 8.0
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
        "name": "web_search",
        "description": "Search Wikipedia for articles matching a query. Returns title, snippet, and article URL.",
        "parameters": {"query": "search terms, e.g. 'python programming'", "limit": "optional max results (default 5)"},
    },
    {
        "name": "read_skill",
        "description": "Read a skill's full SKILL.md by name. Returns the skill file content. Use this to load detailed instructions for an available skill.",
        "parameters": {"name": "skill name, e.g. 'commit'"},
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


def _unified_diff(old_text: str, new_text: str, path: str, max_lines: int = 200, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Return a capped unified diff string."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    diff_iter = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
        n=3,
    )
    diff_lines = list(diff_iter)
    if not diff_lines:
        return ""
    if len(diff_lines) > max_lines:
        remaining = len(diff_lines) - max_lines
        diff_lines = diff_lines[:max_lines]
        diff_lines.append(f"... [truncated {remaining} lines]")
    text = "\n".join(diff_lines)
    if len(text) > max_chars:
        overflow = len(text) - max_chars
        text = text[:max_chars] + f"\n... [truncated {overflow} chars]"
    return text


def _added_lines_diff(content: str, max_lines: int = 200, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Return capped added-lines view (+ prefixed) for new files."""
    lines = content.splitlines()
    capped = lines[:max_lines]
    out = [f"+{ln}" for ln in capped]
    if len(lines) > max_lines:
        out.append(f"... [truncated {len(lines) - max_lines} lines]")
    text = "\n".join(out)
    if len(text) > max_chars:
        overflow = len(text) - max_chars
        text = text[:max_chars] + f"\n... [truncated {overflow} chars]"
    return text


def _retryable(exc: Exception) -> bool:
    """Transient network failures retry; other 4xx fail immediately."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600
    return isinstance(exc, (urllib.error.URLError, OSError))


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
    if len(body) > MAX_OUTPUT_CHARS:
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
    new_text = text.replace(find, replace)
    diff_text = _unified_diff(text, new_text, path)
    p.write_text(new_text, encoding="utf-8")
    if diff_text:
        return f"ok: edited {path}\n{diff_text}"
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
    existed = p.exists()
    p.write_text(content, encoding="utf-8")
    action = "overwrote" if existed else "wrote"
    base = f"ok: {action} {path} ({len(content)} chars)"
    added = _added_lines_diff(content)
    if added:
        return f"{base}\n{added}"
    return base


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


def tool_search(root: Path, query: str, limit: int = 5) -> str:
    if not (query or "").strip():
        return "error: missing_arg: 'query' is required, e.g. {'query': 'python programming'}"
    try:
        lim = int(limit) if limit is not None else 5
    except (TypeError, ValueError):
        return "error: bad_args: 'limit' must be an integer, e.g. {'query': 'python', 'limit': 5}"
    lim = max(1, min(lim, 50))
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(lim),
        "format": "json",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "mini-agent/1.0 (https://example.com; educational)"})
    delay = RETRY_BASE_S
    waited = 0.0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            break
        except Exception as e:
            if not _retryable(e) or waited + delay > RETRY_BUDGET_S:
                return f"error: network failure: {e}"
            time.sleep(delay)
            waited += delay
            delay = min(delay * 2, RETRY_CAP_S)
    try:
        data = json.loads(raw)
    except Exception as e:
        return f"error: parse failure: {e}"
    try:
        results = data.get("query", {}).get("search", [])
    except Exception:
        results = []
    if not results:
        return truncate("no results")
    lines: list[str] = []
    for r in results:
        title = r.get("title", "")
        snippet_html = r.get("snippet", "")
        snippet_text = re.sub(r"<[^>]+>", "", snippet_html)
        snippet_text = html.unescape(snippet_text)
        article_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        lines.append(f"Title: {title}\nSnippet: {snippet_text}\nURL: {article_url}")
    out = "\n\n".join(lines)
    return truncate(out)


# --- Skills support ---

def _parse_skill_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter delimited by ---.

    Returns (metadata dict, body string). If no frontmatter, metadata empty
    and body is original text.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if len(lines) < 3:
        return {}, text
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, text
    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :])
    meta: dict[str, str] = {}
    for ln in fm_lines:
        if not ln.strip():
            continue
        if ":" not in ln:
            continue
        key, val = ln.split(":", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
            val = val[1:-1]
        meta[key] = val
    return meta, body


def load_skills(root: Path) -> dict:
    """Read skills/<name>/SKILL.md at startup.

    Returns dict mapping skill name -> {name, description, content, body, path}.
    Missing skills dir returns empty dict.
    Frontmatter with name and description is parsed; directory name is fallback for name.
    """
    skills_dir = root / "skills"
    if not skills_dir.is_dir():
        return {}
    result: dict = {}
    for entry in skills_dir.iterdir():
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue
        try:
            raw = skill_file.read_text(encoding="utf-8")
        except Exception:
            continue
        meta, body = _parse_skill_frontmatter(raw)
        name = meta.get("name") or entry.name
        desc = meta.get("description") or ""
        if not name:
            continue
        result[name] = {
            "name": name,
            "description": desc,
            "content": raw,
            "body": body,
            "path": str(skill_file.relative_to(root)) if skill_file.is_relative_to(root) else str(skill_file),
        }
    # Also discover any nested SKILL.md files under skills/
    try:
        for p in skills_dir.rglob("SKILL.md"):
            if not p.is_file():
                continue
            try:
                raw = p.read_text(encoding="utf-8")
            except Exception:
                continue
            meta, body = _parse_skill_frontmatter(raw)
            parent = p.parent.name
            name = meta.get("name") or parent
            if name in result:
                continue
            desc = meta.get("description") or ""
            if not name:
                continue
            result[name] = {
                "name": name,
                "description": desc,
                "content": raw,
                "body": body,
                "path": str(p.relative_to(root)) if p.is_relative_to(root) else str(p),
            }
    except Exception:
        pass
    return result


def tool_read_skill(root: Path, name: str) -> str:
    if not (name or "").strip():
        return "error: missing_arg: 'name' is required, e.g. {'name': 'commit'}"
    skills = load_skills(root)
    if name not in skills:
        available = ", ".join(sorted(skills.keys())) if skills else "none"
        return f"error: unknown_skill: {name!r} not found; available skills: {available}"
    skill = skills[name]
    return truncate(skill["content"])


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
    if name == "web_search":
        if "query" not in args or not str(args.get("query", "")).strip():
            return "error: missing_arg: 'query' is required, e.g. {'query': 'python programming'}"
        return tool_search(root, args.get("query", ""), args.get("limit", 5))
    if name == "read_skill":
        if "name" not in args or not str(args.get("name", "")).strip():
            return "error: missing_arg: 'name' is required, e.g. {'name': 'commit'}"
        return tool_read_skill(root, args.get("name", ""))
    if name == "done":
        return "done"
    valid = ", ".join(t["name"] for t in TOOLS)
    return f"error: unknown_tool: {name!r} is not a tool; valid tools are: {valid}"
