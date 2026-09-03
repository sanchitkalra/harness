# Mini coding-agent harness

A ~230-line agent you can read end to end: LLM + tool loop in `agent.py`.

## Setup (project-local env)

```
uv venv
uv pip install -e .      # or: uv sync
source .venv/bin/activate
```

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

## What's inside (learning map)

- `TOOLS` — tool schemas sent to the model
- `resolve()` — sandbox: every path must stay under workspace root
- `tool_read / tool_edit / tool_bash` — with uniqueness check, timeout, truncation
- `run()` — the loop: prompt -> tool call -> execute -> feed back; breaks on `done`, max steps, or repeated calls
- `smoke_test()` + `tests/` — how to test without spending API calls

## Next experiments

1. Add `write_file` tool, watch what breaks (overwrites).
2. Lower `MAX_OUTPUT_CHARS`, see context savings vs lost info.
3. Add per-step approval for `bash` before executing.
4. Log tokens per task; try a cheaper model.
