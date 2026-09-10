"""Rig as an MCP server: exposes one tool, delegate_task, so a bigger coding
harness (e.g. Claude Code) can hand off small, well-scoped tasks to whatever
model Rig is logged into — typically a free/local one via /login's
"compatible" provider — without switching models or spending its own
budget on work that doesn't need it.

Fixed at launch: the workspace (a per-project MCP server config, like any
other local dev-tool MCP server, not a caller-supplied path). Stateless per
call: each delegate_task starts a fresh conversation, no session threading
across the MCP boundary.

There's no human on the other end of an MCP call, so bash approval runs in
"auto" mode with confirm_bash() hard-wired to deny: a risky command the
model flags itself is refused (the model sees why and can retry safely)
rather than either running unsupervised or hanging forever waiting for an
answer nobody can give.

stdout is the MCP wire protocol — nothing here may print to it. The
renderer is a silent no-op rather than ui.PrintRenderer for that reason.

Usage: rig-mcp [--workspace PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import agent


class SilentRenderer:
    """No transcript; see module docstring for why confirm_bash always denies."""

    approval_mode = "auto"

    def confirm_bash(self, command: str) -> bool:
        return False

    def step(self, *a, **k) -> None: pass
    def tool_call(self, *a, **k) -> None: pass
    def tool_result(self, *a, **k) -> None: pass
    def begin_tools(self, *a, **k) -> None: pass
    def end_tools(self, *a, **k) -> None: pass
    def final(self, *a, **k) -> None: pass


def delegate_task(workspace: Path, task: str, max_steps: int = agent.DEFAULT_MAX_STEPS) -> str:
    """Run one task through Rig's own agent loop against workspace, silently."""
    log_path = agent._unique_log_path(workspace)
    return agent.run(task, workspace, max_steps=max_steps, log_path=log_path, renderer=SilentRenderer())


def build_server(workspace: Path) -> FastMCP:
    mcp = FastMCP(name="rig")

    @mcp.tool()
    def delegate_task_tool(task: str, max_steps: int = agent.DEFAULT_MAX_STEPS) -> str:
        """Delegate a small, self-contained coding task to Rig's own agent
        loop and model (often a free/local one) instead of spending your
        own budget on it. Not just drudge work — any well-scoped task with
        a clear, checkable outcome you can fully specify in one instruction:
        read/summarize a file, run the tests and report results, make one
        targeted edit, search the codebase for something specific, and so
        on. Not a fit for open-ended, ambiguous, or judgment-heavy work, or
        anything spanning many files/decisions. Runs inside this server's
        fixed workspace using its own read/edit/write/bash tools. State
        explicitly what the answer/result should be — that's what's
        returned via its `done` call. A bash command it flags as risky is
        refused outright rather than run unsupervised or asked about
        interactively."""
        return delegate_task(workspace, task, max_steps)

    return mcp


def main() -> None:
    ap = argparse.ArgumentParser(description="Rig MCP server")
    ap.add_argument("--workspace", default=".", help="workspace root")
    args = ap.parse_args()
    build_server(Path(args.workspace).resolve()).run(transport="stdio")


if __name__ == "__main__":
    main()
