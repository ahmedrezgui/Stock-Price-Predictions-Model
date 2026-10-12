"""Explicit, one-time owner-authorized initial release; never a replacement gate bypass."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from .api import PredictionRequest, predict_bundle
from .artifacts import load_bundle
from .cloud_storage import immutable, write_json
from .deployment import verify_serving
from .model_registry import bundle_digest, registry_name
from .registry import identity

STATUS = "manual_initial_release_approved"


def authorization(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if value.get("scope") != "initial_release_only" or not value.get("approved_by") or not value.get("reason"):
        raise ValueError("Explicit initial-release authorization required")
    if value.get("production_gate_passed") is not False or not value.get("models"):
        raise ValueError("Manual authorization must not claim a passed production gate")
    return value, sha256(raw).hexdigest()


def reserve(client, path):
    """Consume each model's initial-release permission before any deployment."""
    approval, digest = authorization(path)
    key = f"initial-releases/{digest}.json"
    immutable(client.state, key, Path(path).read_bytes())
    for item in approval["models"]:
        name, version = item["registry_name"], item["registry_version"]
        state = client.get_registered_model(name)
        current = state.aliases.get("deployed")
        if current and (current != version or state.tags.get("initial_approval_sha256") != digest):
            raise ValueError(f"{name} already deployed; replacements require the quality gate")
        entry = client.get_model_version(name, version)
        if entry.tags["bundle_sha256"] != item["bundle_sha256"]:
            raise ValueError("Authorization does not match registered weights")
        # The first reservation is permanent: a different approval/version cannot reuse it.
        reservation = {"approval_sha256": digest, "registry_version": version}
        immutable(client.state, f"initial-releases/used/{name}.json", json.dumps(reservation, sort_keys=True).encode())
    return approval, digest


def package(path, output, models="artifacts/models", data="data/processed"):
    approval, digest = authorization(path)
    prepared = []
    for item in approval["models"]:
        source = Path(models) / item["bundle_version"]
        model, _, metadata = load_bundle(source)
        if registry_name(model.config) != item["registry_name"] or bundle_digest(source) != item["bundle_sha256"]:
            raise ValueError("Authorization model identity/checksum mismatch")
        if model.config.family != "lstm" or item["development_status"] != "candidate":
            raise ValueError("Only development-passing LSTM candidates are authorized")
        request = PredictionRequest(symbol=model.config.symbol, model="lstm", task="return", horizon=1)
        predict_bundle(source, data, request)  # Data validity, freshness and real inference before packaging.
        prepared.append((item, source, model, metadata))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "data").mkdir()
    state = {
        "active": {},
        "history": [],
        "deployment_status": STATUS,
        "initial_approval_sha256": digest,
        "approvals": {},
    }
    for item, source, model, metadata in prepared:
        version = metadata["model_version"]
        shutil.copytree(source, output / "models" / version)
        symbol = model.config.symbol
        shutil.copy2(Path(data) / f"{symbol}.csv", output / "data" / f"{symbol}.csv")
        state["active"][identity(model.config)] = version
        state["approvals"][identity(model.config)] = {"status": STATUS, "approval_id": digest}
    (output / "models/active.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    shutil.copy2(path, output / "initial-release.json")
    return output


def confirm(client, path, endpoint, models="artifacts/models"):
    approval, digest = reserve(client, path)
    # Verify every live model before changing any deployed-version state.
    for item in approval["models"]:
        source = Path(models) / item["bundle_version"]
        _, _, metadata = load_bundle(source)
        if bundle_digest(source) != item["bundle_sha256"]:
            raise ValueError("Local model changed after approval")
        verify_serving(endpoint, metadata, expected_status=STATUS, approval_id=digest)
    for item in approval["models"]:
        name, version = item["registry_name"], item["registry_version"]
        value, etag = client._state(name)
        current = value["aliases"].get("deployed")
        if current and (current != version or value["tags"].get("initial_approval_sha256") != digest):
            raise ValueError("Concurrent deployment detected; initial release cannot replace it")
        value["aliases"]["deployed"] = version
        value["tags"].update(
            deployment_endpoint=endpoint,
            deployment_approval=STATUS,
            initial_approval_sha256=digest,
        )
        write_json(client.state, f"registry/{name}.json", value, etag)
        client.set_model_version_tag(name, version, "deployment_approval", STATUS)
        client.set_model_version_tag(name, version, "initial_approval_sha256", digest)
    receipt = {"approval_sha256": digest, "endpoint": endpoint, "status": STATUS, "models": approval["models"]}
    immutable(client.state, f"initial-releases/{digest}-deployment.json", json.dumps(receipt, sort_keys=True).encode())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["reserve", "package", "confirm"])
    parser.add_argument("--approval", default="configs/initial_release.json")
    parser.add_argument("--mlflow-uri")
    parser.add_argument("--output", default="deployment")
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    if args.action == "package":
        print(package(args.approval, args.output))
        return
    if not args.mlflow_uri or not args.mlflow_uri.startswith("azureml://"):
        parser.error("Azure ML tracking URI required")
    from .azure_registry import registry_client

    client = registry_client(args.mlflow_uri)
    if args.action == "reserve":
        result, digest = reserve(client, args.approval)
        print(f"Reserved {len(result['models'])} exact versions; approval {digest}")
    else:
        if not args.endpoint:
            parser.error("HTTPS endpoint required")
        result = confirm(client, args.approval, args.endpoint)
        print(f"Verified and recorded {len(result['models'])} initial deployments")


if __name__ == "__main__":
    main()
