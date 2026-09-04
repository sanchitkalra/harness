
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
import ui

class RecordingRenderer:
    def __init__(self):
        self.events = []
    def step(self, num, total, text):
        self.events.append(("step", num, total, text))
    def begin_tools(self, step_num):
        self.events.append(("begin_tools", step_num))
    def tool_call(self, num, name, args_json):
        self.events.append(("tool_call", num, name, args_json))
    def tool_result(self, tool_name, text):
        self.events.append(("tool_result", tool_name, text))
    def end_tools(self):
        self.events.append(("end_tools",))
    def final(self, summary):
        self.events.append(("final", summary))


def test_drive_emits_expected_sequence_with_fake_renderer(tmp_path, monkeypatch):
    seq = [
        {
            "content": "thinking about files",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": json.dumps({"path": "a.txt"})}},
                {"id": "c2", "function": {"name": "bash", "arguments": json.dumps({"command": "echo hi"})}},
            ],
        },
        {
            "content": "final thoughts",
            "tool_calls": [
                {"id": "c3", "function": {"name": "done", "arguments": json.dumps({"summary": "finished ok"})}},
            ],
        },
    ]

    def fake_llm_call(messages, tools):
        return seq.pop(0)

    monkeypatch.setattr(agent, "llm_call", fake_llm_call)

    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")

    rec = RecordingRenderer()
    messages = agent.new_conversation("do task", tmp_path)
    result = agent.drive(messages, tmp_path, max_steps=5, renderer=rec)

    assert result == "finished ok"

    steps = [e for e in rec.events if e[0] == "step"]
    assert steps, "no step events recorded"
    assert any("thinking about files" in e[3] for e in steps), f"step text not found in {steps}"
    assert any("final thoughts" in e[3] for e in steps)

    begins = [e for e in rec.events if e[0] == "begin_tools"]
    ends = [e for e in rec.events if e[0] == "end_tools"]
    # original drive() calls end_tools twice on done path (inner + finally) which is idempotent for PrintRenderer
    # so we allow >=
    assert len(begins) == 2, f"expected 2 begin_tools, got {begins}"
    assert len(ends) >= 2, f"expected at least 2 end_tools, got {ends}"
    # ensure each begin is followed by an end eventually (order check)
    # we check that events contain begin before corresponding end
    # simple sanity: first event after step is begin
    # detailed ordering not strict beyond existence

    calls = [e for e in rec.events if e[0] == "tool_call"]
    call_names = [c[2] for c in calls]
    assert "read_file" in call_names
    assert "bash" in call_names
    assert "done" in call_names

    read_calls = [c for c in calls if c[2] == "read_file"]
    assert any("a.txt" in c[3] for c in read_calls)

    results = [e for e in rec.events if e[0] == "tool_result"]
    result_names = [r[1] for r in results]
    assert "read_file" in result_names
    assert "bash" in result_names
    assert any("hello world" in r[2] or "exit=0" in r[2] for r in results)


def test_default_renderer_still_prints(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    import importlib
    importlib.reload(ui)

    seq = [
        {
            "content": "step text hello",
            "tool_calls": [
                {"id": "c1", "function": {"name": "bash", "arguments": json.dumps({"command": "echo hi"})}},
            ],
        },
        {
            "content": "",
            "tool_calls": [
                {"id": "c2", "function": {"name": "done", "arguments": json.dumps({"summary": "done summary"})}},
            ],
        },
    ]

    def fake_llm_call(messages, tools):
        return seq.pop(0)

    monkeypatch.setattr(agent, "llm_call", fake_llm_call)
    import agent as ag_mod
    ag_mod.ui = ui

    messages = ag_mod.new_conversation("task", tmp_path)
    result = ag_mod.drive(messages, tmp_path, max_steps=3)

    out = capsys.readouterr().out
    assert "step text hello" in out or "[step" in out
    assert "bash" in out
    assert result == "done summary"

    capsys.readouterr()
    ui.PrintRenderer().final("some result")
    out2 = capsys.readouterr().out
    assert "some result" in out2
    assert "result" in out2.lower()


def test_renderer_protocol_exists():
    assert hasattr(ui, "Renderer")
    assert hasattr(ui, "PrintRenderer")
    pr = ui.PrintRenderer()
    for method in ("step", "tool_call", "tool_result", "begin_tools", "end_tools", "final"):
        assert hasattr(pr, method), f"PrintRenderer missing {method}"
        assert callable(getattr(pr, method))
