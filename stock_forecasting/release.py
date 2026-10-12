"""Package one exact registered candidate only after a current, passing release gate."""

import argparse
import json
import shutil
from pathlib import Path
from .registry import identity
from .api import PredictionRequest, predict_bundle
from .release_gate import policy_file, verify_gate


def package_release(data, output, client=None, gate_run_id=None, policy_path="configs/release_policy.json"):
    if client is None or not gate_run_id:
        raise ValueError("A passing MLflow release gate is required; local promotion is insufficient")
    policy, _ = policy_file(policy_path)
    data, output = Path(data), Path(output)
    csv = data / f"{policy['symbol']}.csv"
    report, bundle = verify_gate(client, gate_run_id, policy_path, csv)
    source, model, _, metadata, _ = bundle
    request = PredictionRequest(
        symbol=model.config.symbol, model=model.config.family, task=model.config.task, horizon=model.config.horizon
    )
    predict_bundle(source, data, request)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "data").mkdir()
    version = metadata["model_version"]
    shutil.copytree(source, output / "models" / version)
    shutil.copy2(csv, output / "data" / csv.name)
    state = {"active": {identity(model.config): version}, "history": [], "deployment_status": "production_approved"}
    (output / "models/active.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (output / "release.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="deployment")
    parser.add_argument("--mlflow-uri", required=True)
    parser.add_argument("--gate-run-id", required=True)
    parser.add_argument("--policy", default="configs/release_policy.json")
    parser.add_argument("--retain", help="Roster whose other deployed symbols must remain in the image")
    args = parser.parse_args()
    from .azure_registry import registry_client

    client = registry_client(args.mlflow_uri)
    print(package_release(args.data, args.output, client, args.gate_run_id, args.policy))
    if args.retain:
        import os
        from .portfolio import retain_deployed
        from .model_registry import read_gate

        os.environ["FORECAST_DATA_BACKEND"] = "azure"
        retain_deployed(client, args.retain, args.output, read_gate(client, args.gate_run_id)["registry_name"])


if __name__ == "__main__":
    main()
