"""Terminal UI helpers with conditional ANSI colors and collapsing tool batches."""
from __future__ import annotations

import json
import os
import sys

# --- color detection & constants ---

def _use_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

# --- collapsing batch state ---

_batch: list[dict] | None = None
_batch_step: int | None = None
_batch_lines: int = 0


def _is_batch_active() -> bool:
    return _batch is not None


def _println(s: str) -> None:
    """Print a string and count visual lines for collapse."""
    global _batch_lines
    # print() adds one newline; internal \n add more lines
    print(s)
    if _is_batch_active():
        # number of visual lines this print produced
        _batch_lines += s.count("\n") + 1


def _parse_args_json(args_json: str) -> dict:
    try:
        return json.loads(args_json) if args_json else {}
    except Exception:
        return {}


# Public batch API used by agent.py
def begin_tools(step_num: int) -> None:
    global _batch, _batch_step, _batch_lines
    _batch = []
    _batch_step = step_num
    _batch_lines = 0


def end_tools() -> None:
    global _batch, _batch_step, _batch_lines
    if _batch is None:
        return
    batch = _batch
    step_num = _batch_step
    lines = _batch_lines
    # reset early so nested _println doesn't recount
    _batch = None
    _batch_step = None
    _batch_lines = 0

    if not batch:
        return

    # --- compute summary ---
    reads: list[str] = []
    edits: list[str] = []
    writes: list[str] = []
    bashes: list[str] = []
    searches: list[str] = []
    others: dict[str, int] = {}
    errors = 0

    for entry in batch:
        name = entry.get("name") or "?"
        args = entry.get("args") or {}
        result = entry.get("result") or ""
        if isinstance(result, str) and result.lstrip().startswith("error:"):
            errors += 1
        if name == "read_file":
            p = args.get("path")
            if p:
                reads.append(str(p))
            else:
                reads.append("?")
        elif name == "edit_file":
            p = args.get("path")
            if p:
                edits.append(str(p))
            else:
                edits.append("?")
        elif name == "write_file":
            p = args.get("path")
            if p:
                writes.append(str(p))
            else:
                writes.append("?")
        elif name == "bash":
            cmd = args.get("command")
            if cmd:
                bashes.append(str(cmd))
            else:
                bashes.append("?")
        elif name == "web_search":
            q = args.get("query")
            if q:
                searches.append(str(q))
            else:
                searches.append("?")
        elif name == "done":
            continue
        else:
            others[name] = others.get(name, 0) + 1

    parts: list[str] = []


    if reads:
        if len(reads) == 1:
            parts.append(f"read {reads[0]}")
        else:
            # reuse helper but with custom wording
            uniq = []
            seen = set()
            for f in reads:
                if f not in seen:
                    seen.add(f)
                    uniq.append(f)
            preview = uniq[:4]
            if len(reads) > 4:
                parts.append(f"read {len(reads)} files: {', '.join(preview)} +{len(reads)-len(preview)} more")
            else:
                parts.append(f"read {len(reads)} files: {', '.join(preview)}")
    if edits:
        if len(edits) == 1:
            parts.append(f"edited {edits[0]}")
        else:
            uniq = []
            seen = set()
            for f in edits:
                if f not in seen:
                    seen.add(f)
                    uniq.append(f)
            preview = uniq[:4]
            if len(edits) > 4:
                parts.append(f"edited {len(edits)} files: {', '.join(preview)} +{len(edits)-len(preview)} more")
            else:
                parts.append(f"edited {len(edits)} files: {', '.join(preview)}")
    if writes:
        if len(writes) == 1:
            parts.append(f"wrote {writes[0]}")
        else:
            uniq = []
            seen = set()
            for f in writes:
                if f not in seen:
                    seen.add(f)
                    uniq.append(f)
            preview = uniq[:4]
            if len(writes) > 4:
                parts.append(f"wrote {len(writes)} files: {', '.join(preview)} +{len(writes)-len(preview)} more")
            else:
                parts.append(f"wrote {len(writes)} files: {', '.join(preview)}")
    if bashes:
        if len(bashes) == 1:
            parts.append("ran 1 command")
        else:
            parts.append(f"ran {len(bashes)} commands")
    if searches:
        if len(searches) == 1:
            parts.append(f"searched {searches[0]!r}"[:80])
        else:
            parts.append(f"searched {len(searches)} queries")
    for k, v in others.items():
        if v == 1:
            parts.append(k)
        else:
            parts.append(f"{k} x{v}")
    if errors:
        parts.append(f"{errors} error(s)")

    if not parts:
        return

    summary_text = " · ".join(parts)

    # --- render collapsed line ---
    if _use_color():
        # erase the batch's live lines
        if lines > 0:
            # move up and clear each line
            # \033[F = cursor previous line, \033[K = erase to end
            for _ in range(lines):
                sys.stdout.write(f"\033[F\033[K")
            sys.stdout.flush()
        # print collapsed summary
        if errors:
            marker = f"{RED}✖{RESET}"
        else:
            marker = f"{GREEN}✔{RESET}"
        # Use DIM for summary to keep reasoning prominent
        print(f"  {marker} {DIM}↳ {summary_text}{RESET}")
    else:
        # non-TTY / NO_COLOR: keep history but add summary line
        # plain format expected by logs
        s_text = str(step_num) if step_num is not None else "?"
        # keep comma separated for plain
        plain = ", ".join(parts)
        print(f"[step {s_text}] tools: {plain}")


# --- existing UI primitives (now batch-aware) ---

def step(num: int, total: int, text: str) -> None:
    truncated = (text or "")[:300]
    if _use_color():
        print(f"{BOLD}Step {num}/{total}{RESET} {DIM}{truncated}{RESET}")
    else:
        print(f"[step {num}] {truncated}")


def tool_call(num: int, name: str, args_json: str) -> None:
    truncated = (args_json or "")[:200]
    if _is_batch_active():
        args = _parse_args_json(args_json)
        _batch.append({"name": name, "args_json": args_json, "args": args, "result": None})

    if _use_color():
        _println(f"  {CYAN}tool: {name} {truncated}{RESET}")
    else:
        _println(f"[step {num}] tool: {name} {truncated}")


def tool_result(tool_name: str, text: str) -> None:
    raw = text or ""
    stripped = raw.lstrip()
    is_error = stripped.startswith("error:")
    lines = raw.splitlines()
    first = lines[0] if lines else ""
    truncated_first = first[:300]
    truncated_full = raw[:300]

    # update batch entry with result
    if _is_batch_active() and _batch:
        # find most recent entry with same name and no result
        for entry in reversed(_batch):
            if entry.get("name") == tool_name and entry.get("result") is None:
                entry["result"] = raw
                break
        else:
            # no matching pending call (e.g. direct ui.tool_result in tests), append as result-only?
            # don't add new entry to keep summary accurate; just record result-less if needed
            pass

    if _use_color():
        marker = f"{RED}✖{RESET}" if is_error else f"{GREEN}✔{RESET}"
        if tool_name == "read_file":
            _println(f"  {marker} {DIM}{truncated_first}{RESET}")
        elif tool_name in ("edit_file", "write_file"):
            if is_error:
                _println(f"  {marker} {DIM}{truncated_full}{RESET}")
            else:
                _println(f"  {marker} {DIM}{truncated_first}{RESET}")
                for ln in lines[1:]:
                    clipped = ln[:1000]
                    if clipped.startswith("+++") or clipped.startswith("---"):
                        _println(f"  {DIM}{clipped}{RESET}")
                    elif clipped.startswith("@@"):
                        _println(f"  {CYAN}{clipped}{RESET}")
                    elif clipped.startswith("+"):
                        _println(f"  {GREEN}{clipped}{RESET}")
                    elif clipped.startswith("-"):
                        _println(f"  {RED}{clipped}{RESET}")
                    else:
                        _println(f"  {DIM}{clipped}{RESET}")
        else:
            _println(f"  {marker} {DIM}{truncated_full}{RESET}")
    else:
        if tool_name == "read_file":
            _println(f"  -> {truncated_first}")
        elif tool_name in ("edit_file", "write_file"):
            if not lines:
                _println("  ->")
                return
            _println(f"  -> {lines[0][:1000]}")
            for ln in lines[1:1000]:
                _println(f"  {ln[:1000]}")
        else:
            _println(f"  -> {truncated_full}")


def final(summary: str) -> None:
    if _use_color():
        print(f"\n{BOLD}result: {summary}{RESET}")
    else:
        print(f"\nresult: {summary}")


# Backwards compat aliases - if other code imports begin_tool_batch etc
begin_tool_batch = begin_tools
end_tool_batch = end_tools
