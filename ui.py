"""Terminal UI helpers with conditional ANSI colors."""
from __future__ import annotations

import os
import sys


def _use_color() -> bool:
    # Colors only when stdout is a TTY and NO_COLOR is unset
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


def step(num: int, total: int, text: str) -> None:
    truncated = (text or "")[:300]
    if _use_color():
        # Bold header + dimmed model text
        print(f"{BOLD}Step {num}/{total}{RESET} {DIM}{truncated}{RESET}")
    else:
        # Plain matches previous output for logs/pipes
        print(f"[step {num}] {truncated}")


def tool_call(num: int, name: str, args_json: str) -> None:
    truncated = (args_json or "")[:200]
    if _use_color():
        print(f"  {CYAN}tool: {name} {truncated}{RESET}")
    else:
        print(f"[step {num}] tool: {name} {truncated}")


def tool_result(tool_name: str, text: str) -> None:
    raw = text or ""
    stripped = raw.lstrip()
    is_error = stripped.startswith("error:")
    lines = raw.splitlines()
    first = lines[0] if lines else ""
    truncated_first = first[:300]
    truncated_full = raw[:300]

    if _use_color():
        marker = f"{RED}✖{RESET}" if is_error else f"{GREEN}✔{RESET}"
        if tool_name == "read_file":
            # header-only for read
            print(f"  {marker} {DIM}{truncated_first}{RESET}")
        elif tool_name in ("edit_file", "write_file"):
            if is_error:
                print(f"  {marker} {DIM}{truncated_full}{RESET}")
            else:
                print(f"  {marker} {DIM}{truncated_first}{RESET}")
                for ln in lines[1:]:
                    # cap line length to avoid huge lines
                    clipped = ln[:1000]
                    if clipped.startswith("+++") or clipped.startswith("---"):
                        print(f"  {DIM}{clipped}{RESET}")
                    elif clipped.startswith("@@"):
                        print(f"  {CYAN}{clipped}{RESET}")
                    elif clipped.startswith("+"):
                        print(f"  {GREEN}{clipped}{RESET}")
                    elif clipped.startswith("-"):
                        print(f"  {RED}{clipped}{RESET}")
                    else:
                        print(f"  {DIM}{clipped}{RESET}")
        else:
            print(f"  {marker} {DIM}{truncated_full}{RESET}")
    else:
        # NO_COLOR / non-TTY: plain output, preserved for tests/logs
        if tool_name == "read_file":
            print(f"  -> {truncated_first}")
        elif tool_name in ("edit_file", "write_file"):
            if not lines:
                print("  ->")
                return
            print(f"  -> {lines[0][:1000]}")
            for ln in lines[1:1000]:
                print(f"  {ln[:1000]}")
        else:
            print(f"  -> {truncated_full}")


def final(summary: str) -> None:
    if _use_color():
        print(f"\n{BOLD}result: {summary}{RESET}")
    else:
        print(f"\nresult: {summary}")
