"""Unit tests for model_registry.py (persisted login groups + models)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import model_registry as mr


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("RIG_CONFIG_DIR", str(tmp_path))


def test_empty_registry_by_default():
    reg = mr.load_registry()
    assert reg == {"active_group": None, "groups": {}}
    assert mr.get_active_group() is None
    assert mr.get_active_profile() is None


def test_add_group_becomes_active_and_persists():
    mr.add_group("work", "anthropic", "sk-ant-xyz", "claude-sonnet-5")
    reg = mr.load_registry()
    assert reg["active_group"] == "work"
    assert reg["groups"]["work"] == {
        "provider": "anthropic", "api_key": "sk-ant-xyz", "base_url": None,
        "models": ["claude-sonnet-5"], "active_model": "claude-sonnet-5",
    }
    assert mr.get_active_profile() == {
        "provider": "anthropic", "api_key": "sk-ant-xyz", "model": "claude-sonnet-5", "base_url": None,
    }

    # file actually landed on disk with restricted permissions
    p = mr.registry_path()
    assert p.is_file()
    assert (p.stat().st_mode & 0o777) == 0o600


def test_second_group_does_not_steal_active_unless_asked():
    mr.add_group("work", "anthropic", "k1", "claude-sonnet-5")
    mr.add_group("cheap", "openai", "k2", "gpt-4o-mini", make_active=False)
    reg = mr.load_registry()
    assert reg["active_group"] == "work"
    assert set(reg["groups"]) == {"work", "cheap"}


def test_set_active_group_switches_and_rejects_unknown():
    mr.add_group("work", "anthropic", "k1", "claude-sonnet-5")
    mr.add_group("cheap", "openai", "k2", "gpt-4o-mini", make_active=False)
    mr.set_active_group("cheap")
    assert mr.load_registry()["active_group"] == "cheap"
    with pytest.raises(KeyError):
        mr.set_active_group("nope")


def test_remove_group_reassigns_active():
    mr.add_group("work", "anthropic", "k1", "claude-sonnet-5")
    mr.add_group("cheap", "openai", "k2", "gpt-4o-mini", make_active=False)
    mr.remove_group("work")
    reg = mr.load_registry()
    assert "work" not in reg["groups"]
    assert reg["active_group"] == "cheap"


def test_remove_last_group_clears_active():
    mr.add_group("work", "anthropic", "k1", "claude-sonnet-5")
    mr.remove_group("work")
    reg = mr.load_registry()
    assert reg["groups"] == {}
    assert reg["active_group"] is None


def test_add_group_rejects_unknown_provider():
    with pytest.raises(ValueError):
        mr.add_group("x", "cohere", "k", "some-model")


def test_add_model_appends_and_can_switch_active_model():
    mr.add_group("work", "anthropic", "k1", "claude-sonnet-5")
    mr.add_model("work", "claude-opus-5", make_active=False)
    group = mr.get_active_group()
    assert group["models"] == ["claude-sonnet-5", "claude-opus-5"]
    assert group["active_model"] == "claude-sonnet-5"

    mr.set_active_model("work", "claude-opus-5")
    assert mr.get_active_profile()["model"] == "claude-opus-5"

    with pytest.raises(KeyError):
        mr.set_active_model("work", "nope")
    with pytest.raises(KeyError):
        mr.add_model("nope", "some-model")


def test_switching_login_keeps_each_groups_own_active_model():
    mr.add_group("work", "anthropic", "k1", "claude-sonnet-5")
    mr.add_group("cheap", "openai", "k2", "gpt-4o-mini", make_active=False)
    mr.add_model("cheap", "gpt-4o", make_active=True)
    mr.set_active_group("cheap")
    assert mr.get_active_profile()["model"] == "gpt-4o"
    mr.set_active_group("work")
    assert mr.get_active_profile()["model"] == "claude-sonnet-5"


def test_mask_key():
    assert mr.mask_key("sk-ant-1234567890") == "sk-ant...7890"
    assert mr.mask_key("short") == "*****"
    assert mr.mask_key("") == ""


def test_corrupt_registry_file_falls_back_to_empty(tmp_path):
    mr.registry_path().parent.mkdir(parents=True, exist_ok=True)
    mr.registry_path().write_text("not json", encoding="utf-8")
    assert mr.load_registry() == {"active_group": None, "groups": {}}


def test_legacy_flat_profiles_migrate_into_single_model_groups(tmp_path):
    mr.registry_path().parent.mkdir(parents=True, exist_ok=True)
    mr.registry_path().write_text(
        '{"active": "work", "profiles": {"work": '
        '{"provider": "anthropic", "api_key": "k1", "model": "claude-sonnet-5", "base_url": null}}}',
        encoding="utf-8",
    )
    reg = mr.load_registry()
    assert reg["active_group"] == "work"
    assert reg["groups"]["work"]["models"] == ["claude-sonnet-5"]
    assert reg["groups"]["work"]["active_model"] == "claude-sonnet-5"
    assert mr.get_active_profile()["model"] == "claude-sonnet-5"
