"""Azure ML custom-model assets with explicit, conditional Blob deployment state."""

from pathlib import Path
from types import SimpleNamespace
import tempfile

from .cloud_storage import container, read_json, write_json


class AzureRegistry:
    def __init__(self, tracking_uri, ml=None, state=None):
        from mlflow import MlflowClient
        from .azure import ml_client
        import os

        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        self.tracking_uri = tracking_uri
        self.tracking = MlflowClient(tracking_uri=tracking_uri)
        self.ml = ml or ml_client()
        self.state = state or container(os.environ["FORECAST_REGISTRY_CONTAINER"])

    def __getattr__(self, name):
        return getattr(self.tracking, name)

    def _state(self, name):
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return read_json(self.state, f"registry/{name}.json")
        except ResourceNotFoundError:
            return {"aliases": {}, "tags": {}}, None

    def get_registered_model(self, name):
        value, _ = self._state(name)
        return SimpleNamespace(name=name, **value)

    def set_registered_model_alias(self, name, alias, version):
        if alias != "candidate":
            raise ValueError("Azure deployed state requires conditional deployment confirmation")
        value, etag = self._state(name)
        value["aliases"][alias] = str(version)
        write_json(self.state, f"registry/{name}.json", value, etag)

    def confirm(self, name, version, expected, endpoint, gate_run_id):
        value, etag = self._state(name)
        if value["aliases"].get("deployed") != expected:
            raise ValueError("Deployed version changed during release")
        value["aliases"]["deployed"] = str(version)
        value["tags"].update(
            deployment_endpoint=endpoint, deployment_gate_run_id=gate_run_id, deployment_approval="production_approved"
        )
        write_json(self.state, f"registry/{name}.json", value, etag)

    def get_model_version(self, name, version):
        model = self.ml.models.get(name, str(version))
        return SimpleNamespace(
            name=name,
            version=str(model.version),
            status="READY",
            run_id=model.tags["forecast_run_id"],
            tags=model.tags,
            creation_timestamp=int(model.creation_context.created_at.timestamp() * 1000),
        )

    def set_model_version_tag(self, name, version, key, value):
        model = self.ml.models.get(name, str(version))
        model.tags[key] = str(value)
        self.ml.models.create_or_update(model)

    def register_bundle(self, run_id, bundle):
        from azure.ai.ml.entities import Model
        from azure.core.exceptions import ResourceNotFoundError
        from .artifacts import load_bundle
        from .model_registry import bundle_digest, registry_name

        model, _, metadata = load_bundle(bundle)
        run = self.tracking.get_run(run_id)
        if run.info.status != "FINISHED" or run.data.tags.get("workflow") != "training":
            raise ValueError("Completed training provenance required")
        digest = bundle_digest(bundle)
        # Azure ML requires positive integer versions. The full digest below
        # detects collisions instead of silently replacing a different bundle.
        name, version = registry_name(model.config), str(int(digest[:7], 16) + 1)
        artifact = f"models/{Path(bundle).name}"
        with tempfile.TemporaryDirectory() as temp:
            uploaded = self.tracking.download_artifacts(run_id, artifact, temp)
            load_bundle(uploaded)
            if bundle_digest(uploaded) != digest:
                raise ValueError("Uploaded bundle checksum mismatch")
        try:
            existing = self.ml.models.get(name, version)
            if existing.tags["bundle_sha256"] != digest:
                raise ValueError("Model version collision")
        except ResourceNotFoundError:
            self.ml.models.create_or_update(
                Model(
                    name=name,
                    version=version,
                    path=str(bundle),
                    type="custom_model",
                    tags={
                        "forecast_run_id": run_id,
                        "bundle_version": metadata["model_version"],
                        "bundle_sha256": digest,
                        "artifact_path": artifact,
                        "dataset_version": metadata["dataset"]["dataset_version"],
                        "development_status": run.data.tags.get("status", "unknown"),
                        "release_status": "pending",
                    },
                )
            )
        self.set_registered_model_alias(name, "candidate", version)
        return {"name": name, "version": version}


def registry_client(uri):
    from mlflow import MlflowClient

    return AzureRegistry(uri) if uri.startswith("azureml://") else MlflowClient(tracking_uri=uri)
