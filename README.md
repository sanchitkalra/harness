# Mini coding-agent harness

Four small modules with strict boundaries: `agent.py` orchestrates (loop,
sessions, CLI), `model.py` talks to the LLM, `tools.py` touches the
workspace, `ui.py` renders the terminal. Each is readable end to end.

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

Puts `harness` on your PATH (no venv activation needed) so `harness -i` works
from any directory — `--workspace` defaults to `.`, so it targets whichever
project you're standing in.

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

Check the key alone first (no harness involved):

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

In the TUI, type `/model` to open a picker of saved profiles, or add a new
one (name, provider, API key, model). The picked profile becomes active
immediately and is remembered in `~/.config/harness/models.json` (mode 600)
for every future run — no env vars needed. Env vars still override it when
set. Opening `harness -i` with nothing configured at all now works (it used
to hard-exit) — it starts with a tip to run `/model`.

## What's inside (learning map)

- `model.py` — `resolve_provider()` picks a `Provider` strategy (`OpenAIProvider` / `AnthropicProvider`) from env vars, then the saved registry; `llm_call()` calls it. Knows nothing about tools or files. Add a provider by writing a class with `model` + `call(messages, tools)` and wiring it into `resolve_provider()`.
- `model_registry.py` — persisted `(provider, api_key, model)` profiles at `~/.config/harness/models.json`; managed via `/model` in the TUI.
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
