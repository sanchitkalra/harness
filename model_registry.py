"""Persisted model config: named (provider, api_key, model) profiles.

Lets you save credentials once instead of exporting env vars every run.
Storage: ~/.config/harness/models.json, mode 600 (it holds API keys).
Env vars still take precedence in model.resolve_provider() — this is
only consulted when none are set.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROVIDERS = ("anthropic", "openai")


def registry_path() -> Path:
    return Path(os.environ.get("HARNESS_CONFIG_DIR", Path.home() / ".config" / "harness")) / "models.json"


def load_registry() -> dict:
    """{"active": name|None, "profiles": {name: {provider, api_key, model, base_url}}}."""
    p = registry_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("profiles"), dict):
            return {"active": data.get("active"), "profiles": data["profiles"]}
    except Exception:
        pass
    return {"active": None, "profiles": {}}


def save_registry(reg: dict) -> None:
    p = registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    try:
        p.chmod(0o600)
    except Exception:
        pass  # best-effort on platforms without POSIX perms


def add_profile(name: str, provider: str, api_key: str, model: str, base_url: str | None = None, make_active: bool = True) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; must be one of {PROVIDERS}")
    reg = load_registry()
    reg["profiles"][name] = {"provider": provider, "api_key": api_key, "model": model, "base_url": base_url}
    if make_active or not reg.get("active"):
        reg["active"] = name
    save_registry(reg)


def remove_profile(name: str) -> None:
    reg = load_registry()
    reg["profiles"].pop(name, None)
    if reg.get("active") == name:
        reg["active"] = next(iter(reg["profiles"]), None)
    save_registry(reg)


def set_active(name: str) -> None:
    reg = load_registry()
    if name not in reg["profiles"]:
        raise KeyError(f"no saved profile named {name!r}")
    reg["active"] = name
    save_registry(reg)


def get_active_profile() -> dict | None:
    reg = load_registry()
    name = reg.get("active")
    if name is None:
        return None
    return reg["profiles"].get(name)


def mask_key(key: str) -> str:
    key = key or ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"
