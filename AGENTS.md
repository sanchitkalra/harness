# Workspace guide

## Running tests / checks
- Use `.venv/bin/python -m pytest -q` and `.venv/bin/python -m pytest --smoke` etc — plain `python` and `pytest` are NOT on PATH in this environment, use the venv binary explicitly.
- For a quick no-API sanity check: `.venv/bin/python -m agent --smoke --workspace .`
- Bash tool cwd is the workspace root.

## Layout
- `agent.py` — orchestration (run loop, sessions, CLI). Re-exports tools/model names so `agent.*` keeps working.
- `model.py` — LLM client only. Provider strategy pattern: `resolve_provider()` picks an `OpenAIProvider` or `AnthropicProvider` from env vars (`OPENAI_API_KEY` > `ANTHROPIC_API_KEY` > `MODEL_API_KEY`), then falls back to `model_registry`'s active profile; `llm_call()` delegates to it. `llm_config()` still resolves (base, model, key) for OpenAI-compatible providers specifically. Never prints, never touches files.
- `model_registry.py` — persisted named `(provider, api_key, model)` profiles at `~/.config/harness/models.json` (mode 600). No Provider classes here (avoids a model.py <-> model_registry.py import cycle) — model.py reads a profile dict and builds the Provider itself.
- `tools.py` — workspace tools only (`TOOLS`, `resolve`, `tool_*`, `dispatch`). Never prints, never calls the model.
- `ui.py` — terminal rendering only. Takes plain data, never imports model/tools internals.
- `tui.py` — Textual-based interactive TUI (`TuiRenderer`); implements the same `ui.Renderer` protocol plus `read_line`/`update_status`. Renderer methods run on a worker thread and marshal onto the UI thread via `call_from_thread`. Slash commands (`/help`, `/name`, `/clear`, `/model`, `/quit`) are handled entirely inside `read_line()`/`_handle_slash_command()`, never sent to the agent. `/model` pushes `ModelPickerScreen` (a Textual `Select` dropdown) to pick or add a `model_registry` profile — note: don't pass a pre-selected `value=` to that `Select`, mounting with one fires `Select.Changed` immediately and `on_select_changed` would treat it as a real pick and dismiss the screen on open.
- `tests/` — pytest suite; entry `tests/test_agent.py`.
- `sessions/` — JSONL replay logs; managed via `sessions_dir()` which is `root / "sessions"`. Paths logged under `workspace` are absolute/root-canonical.
- `AGENTS.md` takes precedence; `CLAUDE.md` is fallback for workspace instructions via `load_instructions()`.

## Tooling notes
- `read_file` is line-paged (default 200 lines). Use `offset`/`limit` params to page through large files; output tells you next offset.
- `write_file` refuses overwrite unless `overwrite=true` is passed.
- `edit_file` requires unique `find` match.
- `bash` has blocked patterns and timeout; outputs truncated.
- `AGENTS.md`/`MEMORY.md` are injected into the system prompt uncapped (no length limit) — a large `MEMORY.md` grows every call's cost/context, there's no warning for that.
- Model registry file location honors `HARNESS_CONFIG_DIR` env var (tests always set this to a tmp dir — never let a real `~/.config/harness/models.json` leak into a test).
