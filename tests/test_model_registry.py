"""Unit tests for model_registry.py (persisted model config)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import model_registry as mr


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_CONFIG_DIR", str(tmp_path))


def test_empty_registry_by_default():
    reg = mr.load_registry()
    assert reg == {"active": None, "profiles": {}}
    assert mr.get_active_profile() is None


def test_add_profile_becomes_active_and_persists():
    mr.add_profile("work", "anthropic", "sk-ant-xyz", "claude-sonnet-5")
    reg = mr.load_registry()
    assert reg["active"] == "work"
    assert reg["profiles"]["work"] == {
        "provider": "anthropic", "api_key": "sk-ant-xyz", "model": "claude-sonnet-5", "base_url": None,
    }
    assert mr.get_active_profile()["model"] == "claude-sonnet-5"

    # file actually landed on disk with restricted permissions
    p = mr.registry_path()
    assert p.is_file()
    assert (p.stat().st_mode & 0o777) == 0o600


def test_second_profile_does_not_steal_active_unless_asked():
    mr.add_profile("work", "anthropic", "k1", "claude-sonnet-5")
    mr.add_profile("cheap", "openai", "k2", "gpt-4o-mini", make_active=False)
    reg = mr.load_registry()
    assert reg["active"] == "work"
    assert set(reg["profiles"]) == {"work", "cheap"}


def test_set_active_switches_and_rejects_unknown():
    mr.add_profile("work", "anthropic", "k1", "claude-sonnet-5")
    mr.add_profile("cheap", "openai", "k2", "gpt-4o-mini", make_active=False)
    mr.set_active("cheap")
    assert mr.load_registry()["active"] == "cheap"
    with pytest.raises(KeyError):
        mr.set_active("nope")


def test_remove_profile_reassigns_active():
    mr.add_profile("work", "anthropic", "k1", "claude-sonnet-5")
    mr.add_profile("cheap", "openai", "k2", "gpt-4o-mini", make_active=False)
    mr.remove_profile("work")
    reg = mr.load_registry()
    assert "work" not in reg["profiles"]
    assert reg["active"] == "cheap"


def test_remove_last_profile_clears_active():
    mr.add_profile("work", "anthropic", "k1", "claude-sonnet-5")
    mr.remove_profile("work")
    reg = mr.load_registry()
    assert reg["profiles"] == {}
    assert reg["active"] is None


def test_add_profile_rejects_unknown_provider():
    with pytest.raises(ValueError):
        mr.add_profile("x", "cohere", "k", "some-model")


def test_mask_key():
    assert mr.mask_key("sk-ant-1234567890") == "sk-ant...7890"
    assert mr.mask_key("short") == "*****"
    assert mr.mask_key("") == ""


def test_corrupt_registry_file_falls_back_to_empty(tmp_path):
    mr.registry_path().parent.mkdir(parents=True, exist_ok=True)
    mr.registry_path().write_text("not json", encoding="utf-8")
    assert mr.load_registry() == {"active": None, "profiles": {}}
