# Project Memory

- Workspace uses `.venv/bin/python` for tests; plain `python`/`pytest` not on PATH.
- Sessions logs are stored under `sessions/` as JSONL; don't read files under sessions/ per task rules.
- Agent entrypoints: `agent.py` orchestrates, `model.py` is LLM client, `tools.py` defines workspace tools, `ui.py` renders.
