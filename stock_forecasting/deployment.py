"""Verify the serving version before recording a deployment in MLflow."""

import argparse
from datetime import datetime, timezone
import json
import os
from urllib.parse import urlparse

import httpx


def verify_serving(endpoint, metadata):
    url = urlparse(endpoint)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("A trusted HTTPS serving endpoint is required")
    config = metadata["model"]["config"]
    expected = metadata["model_version"]
    headers = {"X-API-Key": os.environ["FORECAST_API_KEY"]}
    # No redirects: do not forward the service key to another host.
    with httpx.Client(timeout=30, follow_redirects=False, headers=headers) as client:
        response = client.get(endpoint.rstrip("/") + "/ready")
        response.raise_for_status()
        if response.json().get("status") != "ready":
            raise ValueError("Deployment is not ready")
        response = client.post(
            endpoint.rstrip("/") + "/v1/predict",
            json={
                "symbol": config["symbol"],
                "model": config["family"],
                "task": config["task"],
                "horizon": config["horizon"],
            },
        )
        response.raise_for_status()
        if response.json().get("model_version") != expected:
            raise ValueError("Serving model differs from the expected registry version")


def confirm_deployment(client, gate_run_id, endpoint, policy_path="configs/release_policy.json"):
    from .release_gate import verify_gate
    from .tracking import TrackingRun

    report, bundle = verify_gate(client, gate_run_id, policy_path)
    verify_serving(endpoint, bundle[3])
    receipt = {
        "gate_run_id": gate_run_id,
        "endpoint": endpoint,
        "registry_name": report["registry_name"],
        "registry_version": report["candidate_version"],
        "bundle_version": bundle[3]["model_version"],
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    TrackingRun(client, gate_run_id).document(receipt, "deployment.json")
    client.set_registered_model_tag(report["registry_name"], "deployment_endpoint", endpoint)
    client.set_registered_model_alias(report["registry_name"], "deployed", report["candidate_version"])
    client.set_model_version_tag(
        report["registry_name"], report["candidate_version"], "deployment_gate_run_id", gate_run_id
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlflow-uri", required=True)
    parser.add_argument("--gate-run-id", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--policy", default="configs/release_policy.json")
    args = parser.parse_args()
    from mlflow import MlflowClient

    print(
        json.dumps(
            confirm_deployment(
                MlflowClient(tracking_uri=args.mlflow_uri), args.gate_run_id, args.endpoint, args.policy
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
