"""Shared candidate training entry point; local by default, Azure submission explicitly gated."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from stock_forecasting.config import Config
from stock_forecasting.training import train_candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--backend", choices=["local", "azure"], default="local")
    parser.add_argument("--data", help="Single local CSV, or Azure dataset URI with --backend azure")
    parser.add_argument("--compute")
    parser.add_argument("--approve-cloud-action", action="store_true")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--artifacts", default="artifacts/models")
    parser.add_argument("--output", default="reports/training")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mlflow-uri", help="Tracking URI; defaults to MLFLOW_TRACKING_URI")
    parser.add_argument("--mlflow-experiment", help="Defaults to MLFLOW_EXPERIMENT_NAME or stock-return-forecasting")
    parser.add_argument(
        "--refit-production",
        action="store_true",
        help="Refit accepted configuration on all matured labels; retire old research holdout",
    )
    args = parser.parse_args(argv)
    config = Config.read(args.config)
    if args.backend == "azure":
        if not args.approve_cloud_action or not args.data or not args.compute:
            parser.error("Azure submission requires --approve-cloud-action, --data and --compute")
        if args.symbols:
            parser.error("Azure single-file training uses the symbol in --config")
        from stock_forecasting.azure import submit_training

        job = submit_training(
            args.config, args.data, args.compute, mlflow_uri=args.mlflow_uri, mlflow_experiment=args.mlflow_experiment
        )
        print(job.name)
        return
    if args.data and args.symbols and len(args.symbols) != 1:
        parser.error("--data supports one symbol")
    results = []
    failed = False
    for symbol in args.symbols or [config.symbol]:
        if (
            not symbol
            or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for ch in symbol)
            or symbol in (".", "..")
        ):
            raise ValueError("Invalid symbol")
        try:
            result = train_candidate(
                Path(args.data) if args.data else Path(args.data_dir) / f"{symbol}.csv",
                replace(config, symbol=symbol),
                args.artifacts,
                args.output,
                args.force,
                args.refit_production,
                args.mlflow_uri,
                args.mlflow_experiment,
            )
        except Exception as exc:
            failed = True
            result = {"symbol": symbol, "status": "failed", "error": str(exc)}
        results.append(result)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "run.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
