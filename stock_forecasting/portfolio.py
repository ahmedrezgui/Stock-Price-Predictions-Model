"""Keep other deployed stocks unchanged when releasing one replacement model."""

import json
from pathlib import Path
import shutil

from .artifacts import load_bundle
from .cloud_storage import refresh_data
from .deployment import verify_serving
from .model_registry import registered_bundle
from .registry import identity


def retain_deployed(client, roster, output, replacing):
    output = Path(output)
    state_path = output / "models/active.json"
    state = json.loads(state_path.read_text())
    retained = []
    for item in json.loads(Path(roster).read_text())["models"]:
        name = item["registry_name"]
        if name == replacing:
            continue
        registered = client.get_registered_model(name)
        version = registered.aliases.get("deployed")
        if not version:
            raise ValueError(f"Missing deployed portfolio member: {name}")
        source, model, _, metadata, _ = registered_bundle(client, name, version)
        shutil.copytree(source, output / "models" / metadata["model_version"])
        refresh_data(output / "data", model.config.symbol)
        if not (output / "data" / f"{model.config.symbol}.csv").exists():
            raise ValueError("Retaining a stock requires its serving data snapshot")
        key = identity(model.config)
        state["active"][key] = metadata["model_version"]
        status = registered.tags.get("deployment_approval", "production_approved")
        state.setdefault("approvals", {})[key] = {
            "status": status,
            "approval_id": registered.tags.get("initial_approval_sha256")
            if status == "manual_initial_release_approved"
            else registered.tags.get("deployment_gate_run_id"),
        }
        retained.append(
            {"registry_name": name, "registry_version": str(version), "bundle_version": metadata["model_version"]}
        )
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    (output / "retained.json").write_text(json.dumps(retained, indent=2), encoding="utf-8")


def verify_retained(client, output, endpoint=None):
    output = Path(output)
    for item in json.loads((output / "retained.json").read_text()):
        current = client.get_registered_model(item["registry_name"]).aliases.get("deployed")
        if str(current) != item["registry_version"]:
            raise ValueError("Another deployed portfolio member changed; rebuild the release")
        if endpoint:
            _, _, metadata = load_bundle(output / "models" / item["bundle_version"])
            verify_serving(endpoint, metadata)
