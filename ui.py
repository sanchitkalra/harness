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


def tool_result(text: str) -> None:
    truncated = (text or "")[:300]
    if _use_color():
        is_error = truncated.lstrip().startswith("error:") or (text or "").lstrip().startswith("error:")
        marker = f"{RED}✖{RESET}" if is_error else f"{GREEN}✔{RESET}"
        print(f"  {marker} {DIM}{truncated}{RESET}")
    else:
        print(f"  -> {truncated}")


def final(summary: str) -> None:
    if _use_color():
        print(f"\n{BOLD}result: {summary}{RESET}")
    else:
        print(f"\nresult: {summary}")
