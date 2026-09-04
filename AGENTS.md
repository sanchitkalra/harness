# Workspace guide

## Running tests / checks
- Use `.venv/bin/python -m pytest -q` and `.venv/bin/python -m pytest --smoke` etc — plain `python` and `pytest` are NOT on PATH in this environment, use the venv binary explicitly.
- For a quick no-API sanity check: `.venv/bin/python -m agent --smoke --workspace .`
- Bash tool cwd is the workspace root.

## Layout
- `agent.py` — main harness (tool loop, sandbox, session logging).
- `tests/` — pytest suite; entry `tests/test_agent.py`.
- `sessions/` — JSONL replay logs; managed via `sessions_dir()` which is `root / "sessions"`. Paths logged under `workspace` are absolute/root-canonical.
- `AGENTS.md` takes precedence; `CLAUDE.md` is fallback for workspace instructions via `load_instructions()`.

## Tooling notes
- `read_file` is line-paged (default 200 lines). Use `offset`/`limit` params to page through large files; output tells you next offset.
- `write_file` refuses overwrite unless `overwrite=true` is passed.
- `edit_file` requires unique `find` match.
- `bash` has blocked patterns and timeout; outputs truncated.
