import argparse
from dataclasses import replace
from pathlib import Path

from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Reproducible return forecasting")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "features", "analyze", "train", "benchmark", "snapshot"):
        command = sub.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--config", default="configs/forecast.json")
        command.add_argument("--output", default=f"reports/{name}")
        if name in ("train", "benchmark"):
            command.add_argument("--family", choices=["lstm", "gru", "arima"])
            command.add_argument("--task", choices=["return", "direction", "multitask"])
            command.add_argument("--epochs", type=int)
            command.add_argument("--horizon", type=int)
            command.add_argument("--seeds", default="7,19,43")
            command.add_argument("--tasks", default="return,direction,multitask")
            command.add_argument("--losses", default="huber,mse")
            command.add_argument("--folds", type=int, default=2)
            command.add_argument("--artifacts", default="artifacts/models")
            command.add_argument("--mlflow-uri")
    report = sub.add_parser("report")
    report.add_argument("--output", required=True)
    sync = sub.add_parser("storage")
    sync.add_argument("action", choices=["upload", "download"])
    sync.add_argument("--file", required=True)
    sync.add_argument("--key", required=True)
    args = parser.parse_args()
    from .reporting import make_reports, write_json

    if args.command == "report":
        make_reports(args.output)
        return
    if args.command == "storage":
        from .storage import configured_store

        store = configured_store()
        if args.action == "upload":
            store.put(args.key, Path(args.file).read_bytes())
        else:
            with Path(args.file).open("xb") as stream:
                stream.write(store.get(args.key))
        return
    c = Config.read(args.config)
    if getattr(args, "epochs", None):
        c = replace(c, epochs=args.epochs)
    if getattr(args, "horizon", None):
        c = replace(c, horizon=args.horizon)
    from .data import features, normalize, read_csv
    from .storage import configured_store, snapshot

    if args.command in ("validate", "snapshot", "features", "analyze"):
        frame, metadata = normalize(read_csv(args.data), c)
        path = Path(args.output)
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "validation.json", metadata)
        if args.command == "snapshot":
            write_json(path / "manifest.json", snapshot(frame, metadata, configured_store()))
        elif args.command == "features":
            engineered, columns = features(frame, c.feature_groups)
            engineered.to_parquet(path / "features.parquet", index=False)
            write_json(path / "feature_columns.json", columns)
        elif args.command == "analyze":
            from .statistics import statistical_analysis

            write_json(
                path / "statistical_analysis.json",
                statistical_analysis(frame.iloc[: int(len(frame) * c.train_fraction)]),
            )
        print(metadata)
        return
    from .benchmark import prepare, run_benchmark, split_metadata

    if args.command == "benchmark":
        result = run_benchmark(
            args.data,
            c,
            args.output,
            tuple(map(int, args.seeds.split(","))),
            tuple(args.tasks.split(",")),
            tuple(args.losses.split(",")),
            args.folds,
            args.artifacts,
            args.mlflow_uri,
        )
        print(f"Scored {len(result['models'])} models. Report: {args.output}/evaluation_report.md")
        return
    from .artifacts import save_bundle
    from .data import fit_scaler, split_masks, transform
    from .models import ARIMAForecaster, create_forecaster

    frame, columns, samples, manifest = prepare(args.data, c, "artifacts/storage")
    end = int(len(frame) * c.train_fraction)
    masks = split_masks(samples, end, int(len(frame) * (c.train_fraction + c.validation_fraction)))
    if args.family == "arima":
        model = ARIMAForecaster().fit(frame.log_return.iloc[1:end].to_numpy())
        path = Path(args.output)
        model.save(
            path,
            {
                "order": model.order,
                "selection": model.selection,
                "dataset": manifest,
                "fit_last_observation": str(frame.timestamp.iloc[end - 1]),
                "horizon": c.horizon,
                "trusted_artifact_only": True,
            },
        )
        print(path)
    else:
        c = replace(c, family=args.family or c.family, task=args.task or c.task)
        scaler, rows = fit_scaler(frame, columns, samples, masks[0], c.sequence_length)
        x = transform(samples.x, scaler)
        model = create_forecaster(len(columns), c).fit(
            x[masks[0]], samples.y[masks[0]], x[masks[1]], samples.y[masks[1]]
        )
        bundle = save_bundle(
            model, scaler, columns, manifest, split_metadata(frame, samples, masks, rows), args.artifacts
        )
        print(bundle)
        import os

        uri = args.mlflow_uri or os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            import mlflow

            mlflow.set_tracking_uri(uri)
            with mlflow.start_run():
                mlflow.log_params({**c.to_dict(), "dataset_version": manifest["dataset_version"]})
                for epoch in model.history:
                    mlflow.log_metrics(
                        {"train_loss": epoch["train_loss"], "validation_loss": epoch["validation_loss"]},
                        step=epoch["epoch"],
                    )
                mlflow.log_artifacts(str(bundle), "model")


if __name__ == "__main__":
    main()
