"""Persisted model config: named auth groups (provider + API key), each
holding one or more model names.

Lets you save credentials once instead of exporting env vars every run,
and flip between models on the same credentials without re-entering a key.
Storage: ~/.config/rig/models.json, mode 600 (it holds API keys).
Env vars still take precedence in model.resolve_provider() — this is
only consulted when none are set.

Shape: {"active_group": name|None, "groups": {name: {provider, api_key,
base_url, models: [name, ...], active_model: name|None}}}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROVIDERS = ("anthropic", "openai", "compatible")


def registry_path() -> Path:
    return Path(os.environ.get("RIG_CONFIG_DIR", Path.home() / ".config" / "rig")) / "models.json"


# ponytail: one-time read fallback for anyone upgrading from the old
# "harness" name — the next save_registry() call migrates them for good.
_LEGACY_PATH = Path.home() / ".config" / "harness" / "models.json"


def _migrate_legacy(data: dict) -> dict:
    """Old shape was {"active": name, "profiles": {name: {provider, api_key,
    model, base_url}}} — one model per group. Fold each profile into its
    own single-model group so existing configs keep working."""
    groups = {}
    for name, p in data.get("profiles", {}).items():
        groups[name] = {
            "provider": p.get("provider"),
            "api_key": p.get("api_key", ""),
            "base_url": p.get("base_url"),
            "models": [p.get("model")] if p.get("model") else [],
            "active_model": p.get("model"),
        }
    return {"active_group": data.get("active"), "groups": groups}


def load_registry() -> dict:
    p = registry_path()
    if not p.is_file() and "RIG_CONFIG_DIR" not in os.environ and _LEGACY_PATH.is_file():
        p = _LEGACY_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"active_group": None, "groups": {}}
    if not isinstance(data, dict):
        return {"active_group": None, "groups": {}}
    if isinstance(data.get("groups"), dict):
        return {"active_group": data.get("active_group"), "groups": data["groups"]}
    if isinstance(data.get("profiles"), dict):
        return _migrate_legacy(data)
    return {"active_group": None, "groups": {}}


def save_registry(reg: dict) -> None:
    p = registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    try:
        p.chmod(0o600)
    except Exception:
        pass  # best-effort on platforms without POSIX perms


def add_group(
    name: str,
    provider: str,
    api_key: str,
    model: str,
    base_url: str | None = None,
    make_active: bool = True,
) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; must be one of {PROVIDERS}")
    reg = load_registry()
    reg["groups"][name] = {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "models": [model],
        "active_model": model,
    }
    if make_active or not reg.get("active_group"):
        reg["active_group"] = name
    save_registry(reg)


def remove_group(name: str) -> None:
    reg = load_registry()
    reg["groups"].pop(name, None)
    if reg.get("active_group") == name:
        reg["active_group"] = next(iter(reg["groups"]), None)
    save_registry(reg)


def set_active_group(name: str) -> None:
    reg = load_registry()
    if name not in reg["groups"]:
        raise KeyError(f"no saved login named {name!r}")
    reg["active_group"] = name
    save_registry(reg)


def get_active_group() -> dict | None:
    """Active group's config plus its own "name" key, or None."""
    reg = load_registry()
    name = reg.get("active_group")
    if name is None or name not in reg["groups"]:
        return None
    return {"name": name, **reg["groups"][name]}


def add_model(group_name: str, model: str, make_active: bool = True) -> None:
    reg = load_registry()
    if group_name not in reg["groups"]:
        raise KeyError(f"no saved login named {group_name!r}")
    group = reg["groups"][group_name]
    if model not in group["models"]:
        group["models"].append(model)
    if make_active or not group.get("active_model"):
        group["active_model"] = model
    save_registry(reg)


def set_active_model(group_name: str, model: str) -> None:
    reg = load_registry()
    if group_name not in reg["groups"]:
        raise KeyError(f"no saved login named {group_name!r}")
    group = reg["groups"][group_name]
    if model not in group["models"]:
        raise KeyError(f"no model {model!r} saved under login {group_name!r}")
    group["active_model"] = model
    save_registry(reg)


def get_active_profile() -> dict | None:
    """Flat {provider, api_key, model, base_url} for whichever group/model
    is active — the shape model.py's resolve_provider() consumes."""
    group = get_active_group()
    if group is None or not group.get("active_model"):
        return None
    return {
        "provider": group["provider"],
        "api_key": group["api_key"],
        "model": group["active_model"],
        "base_url": group.get("base_url"),
    }


def mask_key(key: str) -> str:
    key = key or ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"
