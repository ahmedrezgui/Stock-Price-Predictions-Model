"""Local version selection: immutable bundles, atomic active manifest, auditable rollback."""

import json
import os
import uuid
from pathlib import Path
from .artifacts import load_bundle


def identity(config):
    return f"{config.symbol}-{config.family}-{config.task}-h{config.horizon}"


def read_registry(root):
    path = Path(root) / "active.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"active": {}, "history": []}


def promote(root, version, expected=None):
    root = Path(root)
    if Path(version).name != version or version in (".", ".."):
        raise ValueError("Invalid model version")
    model, _, _ = load_bundle(root / version)
    key = identity(model.config)
    if expected is not None and key != expected:
        raise ValueError("Model identity mismatch")
    state = read_registry(root)
    previous = state["active"].get(key)
    state["active"][key] = version
    state["history"].append({"identity": key, "previous": previous, "version": version})
    temporary = root / f".active-{uuid.uuid4().hex}.json"
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, root / "active.json")
    return state


def rollback(root, key):
    state = read_registry(root)
    current = state["active"].get(key)
    for entry in reversed(state["history"]):
        if entry["identity"] == key and entry["version"] == current and entry["previous"]:
            return promote(root, entry["previous"], key)
    raise ValueError("No previous version for this identity")
