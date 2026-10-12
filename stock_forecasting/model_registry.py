"""MLflow registry of verified custom bundles; aliases never change during training except candidate."""

from hashlib import sha256
import json
from pathlib import Path

from .artifacts import load_bundle


def registry_name(config):
    # Architecture is deliberately absent: LSTM and GRU compete for the same service.
    return f"stock-{config.symbol}-{config.task}-h{config.horizon}"


def bundle_digest(path):
    return sha256((Path(path) / "checksums.json").read_bytes()).hexdigest()


def register_candidate(client, run_id, bundle):
    """Register uploaded weights, never a benchmark run containing only metrics."""
    bundle = Path(bundle)
    model, _, metadata = load_bundle(bundle)
    run = client.get_run(run_id)
    if run.info.status != "FINISHED" or run.data.tags.get("workflow") != "training":
        raise ValueError("A completed routine training run is required")
    name = registry_name(model.config)
    from mlflow.exceptions import MlflowException

    try:
        client.get_registered_model(name)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        client.create_registered_model(
            name, description="Verified forecasting bundles; deployment requires a release gate."
        )
    artifact = f"models/{bundle.name}"
    # Verify the uploaded copy as well as the local source before publishing its registry entry.
    uploaded = Path(client.download_artifacts(run_id, artifact))
    load_bundle(uploaded)
    if bundle_digest(uploaded) != bundle_digest(bundle):
        raise ValueError("Uploaded bundle differs from local candidate")
    for existing in client.search_model_versions(f"name = '{name}'"):
        if existing.tags.get("bundle_version") == bundle.name and existing.tags.get("bundle_sha256") == bundle_digest(
            bundle
        ):
            client.set_registered_model_alias(name, "candidate", existing.version)
            return {"name": name, "version": str(existing.version)}
    version = client.create_model_version(
        name,
        source=f"runs:/{run_id}/{artifact}",
        run_id=run_id,
        tags={
            "bundle_version": bundle.name,
            "bundle_sha256": bundle_digest(bundle),
            "artifact_path": artifact,
            "dataset_version": metadata["dataset"]["dataset_version"],
            "development_status": run.data.tags.get("status", "unknown"),
            "release_status": "pending",
        },
    )
    client.set_registered_model_alias(name, "candidate", version.version)
    return {"name": name, "version": str(version.version)}


def deployed_version(client, name):
    # A missing alias is different from an unreachable registry; errors must not imply bootstrap.
    return client.get_registered_model(name).aliases.get("deployed")


def registered_bundle(client, name, version, destination=None):
    entry = client.get_model_version(name, str(version))
    if entry.status != "READY" or not entry.run_id:
        raise ValueError("Registry version is not ready or lacks training lineage")
    run = client.get_run(entry.run_id)
    if run.info.status != "FINISHED" or run.data.tags.get("workflow") != "training":
        raise ValueError("Registry version does not reference completed routine training")
    bundle_version = entry.tags["bundle_version"]
    if Path(bundle_version).name != bundle_version or bundle_version in (".", ".."):
        raise ValueError("Invalid bundle version")
    artifact = f"models/{bundle_version}"
    if entry.tags["artifact_path"] != artifact:
        raise ValueError("Registry artifact path mismatch")
    path = Path(client.download_artifacts(entry.run_id, artifact, dst_path=str(destination) if destination else None))
    model, scaler, metadata = load_bundle(path)
    if registry_name(model.config) != name or metadata["model_version"] != bundle_version:
        raise ValueError("Registered model identity mismatch")
    if bundle_digest(path) != entry.tags["bundle_sha256"]:
        raise ValueError("Registered bundle checksum changed")
    return path, model, scaler, metadata, entry


def read_gate(client, run_id):
    run = client.get_run(run_id)
    if run.info.status != "FINISHED" or run.data.tags.get("workflow") != "release_gate":
        raise ValueError("A completed release-gate run is required")
    path = client.download_artifacts(run_id, "metadata/gate.json")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    if report.get("gate_run_id") != run_id or digest != run.data.params.get("gate_evidence_sha256"):
        raise ValueError("Gate evidence was modified or belongs to a different run")
    return report
