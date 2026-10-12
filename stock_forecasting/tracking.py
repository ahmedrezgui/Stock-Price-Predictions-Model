"""Explicit MLflow runs shared by notebooks and training; no global active-run state."""

from contextlib import contextmanager
from hashlib import sha256
import importlib.metadata
import json
import math
from numbers import Real
import os
from pathlib import Path
import platform
import tempfile

from .artifacts import code_version
from .data import FEATURE_VERSION


def finite_metrics(values, prefix=""):
    """Flatten numeric scalars; undefined metrics and arrays stay in JSON artifacts."""
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(finite_metrics(value, name))
        elif isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
            result[name] = float(value)
    return result


class TrackingRun:
    def __init__(self, client=None, run_id=None, experiment_id=None):
        self.client, self.run_id, self.experiment_id = client, run_id, experiment_id

    def params(self, values):
        if self.client:
            from mlflow.entities import Param

            self.client.log_batch(
                self.run_id,
                params=[
                    Param(str(k), json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else str(v))
                    for k, v in values.items()
                ],
            )

    def metrics(self, values, prefix="", step=0):
        if self.client:
            from mlflow.entities import Metric
            import time

            metrics = finite_metrics(values, prefix)
            self.client.log_batch(
                self.run_id, metrics=[Metric(k, v, int(time.time() * 1000), int(step)) for k, v in metrics.items()]
            )

    def tags(self, values):
        if self.client:
            from mlflow.entities import RunTag

            self.client.log_batch(self.run_id, tags=[RunTag(str(key), str(value)) for key, value in values.items()])

    def document(self, values, name):
        if self.client:
            # Temporary files do not add JSON clutter to the user's reports directory.
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / name
                path.write_text(json.dumps(values, indent=2, default=str, allow_nan=False), encoding="utf-8")
                self.client.log_artifact(self.run_id, str(path), "metadata")

    def artifacts(self, path, destination):
        if self.client:
            path = Path(path)
            if path.is_dir():
                self.client.log_artifacts(self.run_id, str(path), destination)
            else:
                self.client.log_artifact(self.run_id, str(path), destination)

    def dataset(self, manifest, split, columns):
        self.params(
            {
                "dataset_version": manifest["dataset_version"],
                "source_file_sha256": manifest["source_file_sha256"],
                "feature_version": FEATURE_VERSION,
            }
        )
        self.document(manifest, "dataset.json")
        self.document({"split": split, "feature_columns": columns}, "inputs.json")

    def history(self, history):
        self.metrics({"epochs_run": len(history)}, "training")
        if history and all("validation_loss" in epoch for epoch in history):
            best = min(history, key=lambda epoch: epoch["validation_loss"])
            self.metrics({"best_epoch": best["epoch"]}, "training")
        if self.client:
            from mlflow.entities import Metric
            import time

            timestamp = int(time.time() * 1000)
            metrics = [
                Metric(key, value, timestamp, int(epoch["epoch"]))
                for epoch in history
                for key, value in finite_metrics({k: v for k, v in epoch.items() if k != "epoch"}, "training").items()
            ]
            # Keep every epoch while avoiding one network round trip per epoch.
            for start in range(0, len(metrics), 1000):
                self.client.log_batch(self.run_id, metrics=metrics[start : start + 1000])
        self.document(history, "history.json")


@contextmanager
def tracking_run(config, workflow, uri=None, experiment=None, parent=None, name=None, family=None):
    """Disabled unless a URI/environment is configured. Enabled tracking fails loudly."""
    uri = uri or os.getenv("MLFLOW_TRACKING_URI")
    if not uri and not (parent and parent.client):
        yield TrackingRun()
        return
    try:
        from mlflow import MlflowClient
    except ImportError as exc:
        raise RuntimeError('MLflow tracking requires: python -m pip install -e ".[tracking]"') from exc

    client = parent.client if parent and parent.client else MlflowClient(tracking_uri=uri)
    if parent and parent.client:
        experiment_id = parent.experiment_id
    else:
        experiment = experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "stock-return-forecasting")
        existing = client.get_experiment_by_name(experiment)
        if existing is None:
            try:
                location = None
                if uri and uri.startswith("sqlite:///"):
                    folder = Path(uri.removeprefix("sqlite:///")).resolve().parent
                    location = (folder / "mlruns" / sha256(experiment.encode()).hexdigest()[:12]).as_uri()
                experiment_id = client.create_experiment(experiment, artifact_location=location)
            except Exception:
                existing = client.get_experiment_by_name(experiment)
                if existing is None:
                    raise
                experiment_id = existing.experiment_id
        else:
            experiment_id = existing.experiment_id
    tags = {
        "workflow": workflow,
        "tracking_schema": "1",
        "mlflow.runName": name or f"{workflow}-{config.symbol}-{config.family}-{config.task}",
    }
    if parent and parent.client:
        tags["mlflow.parentRunId"] = parent.run_id
    run = TrackingRun(client, client.create_run(experiment_id, tags=tags).info.run_id, experiment_id)
    try:
        settings = config.to_dict()
        if family:
            settings["family"] = family
        run.params(settings)
        run.tags(code_version())
        environment = {"python": platform.python_version()}
        for package in ("mlflow", "torch", "numpy", "pandas", "pyarrow", "scikit-learn", "statsmodels"):
            environment[package] = importlib.metadata.version(package)
        run.document(environment, "environment.json")
        run.document(settings, "config.json")
        yield run
    except BaseException:
        client.set_terminated(run.run_id, "FAILED")
        raise
    else:
        client.set_terminated(run.run_id, "FINISHED")
