"""Explicit cloud preparation: migrate existing candidates, publish data and reconcile forecasts."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .artifacts import load_bundle
from .azure_registry import registry_client
from .cloud_storage import publish_data
from .market import write_json
from .model_registry import register_candidate
from .tracking import tracking_run
from .training import log_candidate


def migrate(result, root, uri):
    """New import run, original immutable bundle/evidence; never claim retraining."""
    bundle = Path(root) / result["model_version"]
    model, _, metadata = load_bundle(bundle)
    with tracking_run(model.config, "training", uri, name=f"import-{model.config.symbol}-{bundle.name}") as run:
        run.tags(
            {
                "imported": True,
                "source_run_id": result["mlflow_run_id"],
                "source_model_created_at": metadata["created_at"],
            }
        )
        run.document(result, "original_candidate.json")
        log_candidate(run, result, root)
    registered = register_candidate(registry_client(uri), run.run_id, bundle)
    return {
        **registered,
        "azure_run_id": run.run_id,
        "bundle_version": bundle.name,
        "source_run_id": result["mlflow_run_id"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["migrate", "publish", "monitor"])
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--roster", help="Read serving symbols from an initial-release manifest")
    parser.add_argument("--results", default="artifacts/market/results.json")
    parser.add_argument("--models", default="artifacts/models")
    parser.add_argument("--data")
    parser.add_argument("--container", default="forecast-serving")
    parser.add_argument("--mlflow-uri")
    args = parser.parse_args()
    if args.roster:
        if args.symbols:
            parser.error("Use --symbols or --roster, not both")
        args.symbols = [entry["symbol"] for entry in json.loads(Path(args.roster).read_text())["models"]]
    if args.mlflow_uri:
        # Azure's artifact plugin also consults the process-wide tracking URI.
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    if args.action == "monitor":
        from .cloud_storage import reconcile_remote, refresh_data

        if not args.symbols:
            parser.error("Monitoring requires explicit --symbols")
        cache = args.data or "artifacts/cloud-data"
        os.environ["FORECAST_DATA_BACKEND"] = "azure"
        os.environ["FORECAST_MONITOR_BACKEND"] = "azure"
        for symbol in args.symbols:
            refresh_data(cache, symbol)
        result = reconcile_remote(cache, "reports/monitoring/monitoring_report.md", uri=args.mlflow_uri)
        print(f"Forecasts: {result['forecast_count']}; excluded data issues: {len(result['issues'])}")
        if result["issues"]:
            raise SystemExit(1)
        return
    args.data = args.data or "data/processed"
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    selected = args.symbols or [s for s, r in results.items() if r["status"] == "candidate"]
    for symbol in selected:
        if results[symbol]["validation_objective"] > results[symbol]["baseline_objective"]:
            raise ValueError(f"{symbol} did not pass development validation")
    index_path = Path("artifacts/azure/candidates.json")
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for symbol in selected:
        if args.action == "migrate":
            if not args.mlflow_uri or not args.mlflow_uri.startswith("azureml://"):
                parser.error("Azure ML tracking URI required")
            # Re-running migration may create another import run, but reuses the immutable asset version.
            index[symbol] = migrate(results[symbol], args.models, args.mlflow_uri)
            write_json(index_path, index)
            policy_path = Path("configs/release_policies") / f"{symbol}.json"
            if not policy_path.exists():
                policy = json.loads(Path("configs/release_policy.json").read_text())
                policy.update(symbol=symbol, frozen_at=datetime.now(timezone.utc).isoformat())
                write_json(policy_path, policy)
            print(f"Registered {symbol}: {index[symbol]['version']}", flush=True)
        elif args.action == "publish":
            print(symbol, publish_data(Path(args.data) / f"{symbol}.csv", symbol, args.container), flush=True)


if __name__ == "__main__":
    main()
