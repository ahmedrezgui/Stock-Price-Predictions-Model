"""Dedicated MLflow GenAI experiment, separate from model-training runs."""

import os
from pathlib import Path
from threading import Lock

_lock = Lock()
_configured = None


def configure_tracking():
    global _configured
    if os.getenv("ASSISTANT_TRACING_ENABLED", "true").lower() == "false":
        return None
    uri = os.getenv("ASSISTANT_MLFLOW_TRACKING_URI")
    uri = uri or "sqlite:///" + (Path(__file__).resolve().parents[1] / "mlflow-assistant.db").as_posix()
    name = os.getenv("ASSISTANT_MLFLOW_EXPERIMENT", "stock-assistant")
    with _lock:
        if _configured == (uri, name):
            return name
        import mlflow
        import mlflow.langchain

        mlflow.set_tracking_uri(uri)
        experiment = mlflow.set_experiment(name)
        mlflow.MlflowClient().set_experiment_tag(experiment.experiment_id, "mlflow.experimentKind", "genai_development")
        mlflow.langchain.autolog(log_traces=True, silent=False)
        _configured = (uri, name)
    return name
