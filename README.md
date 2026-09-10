# Rig

A minimal coding-agent harness. Four small modules with strict boundaries:
`agent.py` orchestrates (loop, sessions, CLI), `model.py` talks to the LLM,
`tools.py` touches the workspace, `ui.py` renders the terminal. Each is
readable end to end.

## Setup (project-local env)

```
uv venv
uv pip install -e .      # or: uv sync
source .venv/bin/activate
```

## Install as a global command

```
uv tool install --editable .
```

Puts `rig` on your PATH (no venv activation needed) so `rig` works from any
directory — interactive mode is on by default (no more typing `-i` every
time), and `--workspace` defaults to `.`, so it targets whichever project
you're standing in.

## Try it without an API key

```
python -m agent --smoke --workspace .
python -m pytest -q
```

## Try it with Muse Spark

Get a key at [dev.meta.ai](https://dev.meta.ai) (dashboard → API keys → Create API key), then:

```
export MODEL_API_KEY=...
# optional: export MUSE_SPARK_MODEL=muse-spark-1.1 MUSE_SPARK_BASE_URL=https://api.meta.ai/v1
python -m agent "create hello.txt with one line, then verify with ls" --workspace .
cat sessions/*.jsonl  # replay log
```

Check the key alone first (no rig involved):

```
curl -s https://api.meta.ai/v1/models -H "Authorization: Bearer $MODEL_API_KEY" | head -c 500
```

OpenAI works too: set `OPENAI_API_KEY` (takes precedence) with optional `OPENAI_BASE_URL` / `OPENAI_MODEL`.

Anthropic works too: set `ANTHROPIC_API_KEY` (used when `OPENAI_API_KEY` is unset) with optional
`ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` (defaults to `claude-sonnet-4-5`) / `ANTHROPIC_MAX_TOKENS`
(defaults to 8192 — raise it if a task needs to generate a lot of output in one tool call, e.g.
writing a long file; too small a budget truncates the tool call's JSON before it finishes).

Provider precedence: `OPENAI_API_KEY` > `ANTHROPIC_API_KEY` > `MODEL_API_KEY` (Muse Spark)
> the saved registry (below).

## Save model config instead of exporting env vars every run

In the TUI, type `/login` to open a picker of saved logins (name, provider —
anthropic/openai/compatible —, API key, base URL) or add a new one. `/model`
then picks or adds a model within the active login, so switching models
doesn't require re-entering credentials. Both are remembered in
`~/.config/rig/models.json` (mode 600) for every future run — no env
vars needed. Env vars still override it when set. Opening `rig` with
nothing configured at all now works (it used to hard-exit) — it starts with
a tip to run `/login`.

## Bash approval modes

The TUI gates `bash` calls behind one of three modes, shown in the footer
and cycled with **shift+tab** (or set explicitly with `/mode plan|auto|yolo`):

- **plan** — every bash command pauses for your approval first.
- **auto** (default) — the model flags a call `risk="confirm"` when it's
  destructive or hard to undo (rm, force-push, migrations, installs,
  network writes); only those pause for approval, routine commands run
  immediately.
- **yolo** — nothing is ever confirmed.

A denied command is reported back to the model as an error so it can try a
different approach instead of getting stuck. This only applies to the TUI —
plain/one-shot runs (`--no-tui`, `--no-interactive`) always behave like yolo
since there's no one there to ask.

## Rig as an MCP server (delegate small tasks from a bigger harness)

`rig-mcp --workspace .` runs Rig as an MCP stdio server exposing one tool,
`delegate_task_tool(task, max_steps?)`, so a bigger coding harness (e.g.
Claude Code) can hand off small, well-scoped tasks — not just drudge work,
anything with a clear, checkable outcome you can fully specify in one
instruction: read/summarize a file, run the tests, make a targeted edit,
search the codebase for something specific — to whichever model *you've*
configured Rig to use, typically a free/local one via `/login`'s
`compatible` provider, without switching models or spending its own budget
on work that doesn't need it. Point your harness's MCP config at `rig-mcp`
with `--workspace <project>` args, per-project like any other local
dev-tool MCP server.

Each call is stateless (a fresh conversation) and silent (no transcript —
stdout is the wire protocol). There's no human on the other end of an MCP
call, so bash approval runs in `auto` mode with a twist: a command the model
flags `risk="confirm"` is refused outright rather than run unsupervised or
left hanging waiting for an approval nobody can give — the model sees the
denial and can retry a safer way.

## What's inside (learning map)

- `model.py` — `resolve_provider()` picks a `Provider` strategy (`OpenAIProvider` / `AnthropicProvider`) from env vars, then the saved registry; `llm_call()` calls it. Knows nothing about tools or files. Add a provider by writing a class with `model` + `call(messages, tools)` and wiring it into `resolve_provider()`.
- `model_registry.py` — persisted logins (`provider`, `api_key`, `base_url`, and the models saved under them) at `~/.config/rig/models.json`; logins managed via `/login`, models within the active login via `/model` in the TUI.
- `tools.py` — `TOOLS`, `resolve()` sandbox, `tool_read / tool_edit / tool_write / tool_bash`, `dispatch()`
- `ui.py` — terminal rendering (ANSI on tty only); takes plain data, never imports the other modules
- `agent.py` — `run()` loop, session headers + `--list` / `--fork`, `load_instructions()`, `smoke_test()`, CLI
- `tui.py` — `TuiRenderer`, a Textual app implementing `ui.Renderer` for interactive mode (`--no-tui` to fall back to plain prints)
- `smoke_test()` + `tests/` — how to test without spending API calls

Coupling rules: agent talks to the model only via `llm_call(messages, tools)`,
to tools only via `dispatch(name, args)`, to UI only via `ui.*`. Tests import
`agent.*` (re-exported) so module moves don't break them.

## Next experiments

1. Add `write_file` tool, watch what breaks (overwrites).
2. Lower `MAX_OUTPUT_CHARS`, see context savings vs lost info.
3. Add per-step approval for `bash` before executing.
4. Log tokens per task; try a cheaper model.
