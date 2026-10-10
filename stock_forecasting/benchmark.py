"""Validation-only selection, development walk-forward folds, final holdout."""

from dataclasses import replace
from hashlib import sha256
import importlib.metadata
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .artifacts import code_version, save_bundle
from .backtesting import backtest
from .config import Config
from .data import FEATURE_VERSION, features, fit_scaler, normalize, read_csv, sequences, split_masks, targets, transform
from .evaluation import block_interval, correlation, diebold_mariano, evaluate
from .models import ARIMAForecaster, create_forecaster, naive_predictions
from .reporting import make_reports, write_json
from .statistics import statistical_analysis
from .storage import LocalStore, snapshot
from .tracking import tracking_run


def prepare(csv, config, storage_root):
    frame, metadata = normalize(read_csv(csv), config)
    metadata["source_file_sha256"] = sha256(Path(csv).read_bytes()).hexdigest()
    manifest = snapshot(frame, metadata, LocalStore(storage_root))
    frame, columns = features(frame, config.feature_groups)
    frame = targets(frame, config.horizon)
    samples = sequences(frame, columns, config.sequence_length, config.horizon)
    return frame, columns, samples, manifest


def split_metadata(frame, samples, masks, scaler_rows):
    result = {
        "purge": "label end strictly before next split origin",
        "scaler_last_input": str(frame.timestamp.iloc[scaler_rows[-1]]),
    }
    for name, mask in zip(("train", "validation", "test"), masks):
        result[f"{name}_start"] = str(frame.timestamp.iloc[samples.origins[mask][0]])
        result[f"{name}_end"] = str(frame.timestamp.iloc[samples.origins[mask][-1]])
        result[f"{name}_last_label"] = str(frame.timestamp.iloc[samples.ends[mask][-1]])
        result[f"{name}_n"] = int(mask.sum())
    return result


def objective(y, prediction):
    if prediction["return"] is not None:
        return float(np.mean((prediction["return"] - y) ** 2))
    return float(np.mean((prediction["probability"] - (y > 0)) ** 2))


def run_benchmark(
    csv: str,
    config: Config,
    output: str,
    seeds=(7, 19, 43),
    tasks=("return", "direction", "multitask"),
    losses=("huber", "mse"),
    folds=2,
    artifacts="artifacts/models",
    mlflow_uri=None,
    mlflow_experiment=None,
):
    with tracking_run(config, "research_benchmark", mlflow_uri, mlflow_experiment) as tracking:
        return _run_benchmark(csv, config, output, seeds, tasks, losses, folds, artifacts, tracking)


def _run_benchmark(csv, config, output, seeds, tasks, losses, folds, artifacts, tracking):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "curves").mkdir()
    frame, columns, samples, manifest = prepare(csv, config, "artifacts/storage")
    train_end = int(len(frame) * config.train_fraction)
    validation_end = int(len(frame) * (config.train_fraction + config.validation_fraction))
    masks = split_masks(samples, train_end, validation_end)
    train, val, test = masks
    scaler, scaler_rows = fit_scaler(frame, columns, samples, train, config.sequence_length)
    x = transform(samples.x, scaler)
    summary = {
        "dataset": manifest,
        "feature_version": FEATURE_VERSION,
        "config": config.to_dict(),
        "split": split_metadata(frame, samples, masks, scaler_rows),
        "seeds": list(seeds),
        "models": {},
        "tuning": [],
        "walk_forward": [],
        "selection": {},
        **code_version(),
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "scikit-learn", "statsmodels")
        },
    }
    frame.to_csv(root / "history.csv", index=False)
    write_json(root / "statistical_analysis.json", statistical_analysis(frame.iloc[:train_end]))
    predictions = []
    train_y, y = samples.y[train], samples.y[test]
    origins = samples.origins[test]
    volatility_cut = float(frame.log_return.iloc[:train_end].rolling(20).std().median())

    def record(name, pred, extra=None):
        metric = evaluate(y, pred, train_y)
        result = {"metrics": metric, **(extra or {})}
        signal = (
            pred["return"]
            if pred.get("return") is not None
            else (
                pred["probability"] - 0.5
                if pred.get("probability") is not None
                else np.asarray(pred["direction"], dtype=float) - 0.5
            )
        )
        result["backtest"], curve, trades = backtest(
            frame, origins, signal, config.horizon, config.threshold, config.cost_bps
        )
        curve.to_csv(root / "curves" / f"{name}.csv", index=False)
        write_json(root / "curves" / f"{name}_trades.json", trades)
        direction = pred.get("direction")
        if direction is None:
            direction = pred["probability"] >= 0.5 if pred.get("probability") is not None else pred["return"] > 0
        result["direction_accuracy_95_block_ci"] = block_interval((y > 0) == direction, config.horizon)
        if pred.get("return") is not None:
            result["mae_95_block_ci"] = block_interval(abs(y - pred["return"]), config.horizon)
            result["dm_vs_zero"] = diebold_mariano(y, pred["return"], np.zeros(len(y)), config.horizon)
            rolling = [correlation(y[i - 59 : i + 1], pred["return"][i - 59 : i + 1], True) for i in range(59, len(y))]
            valid = [v for v in rolling if v is not None]
            result["rolling_ic"] = {
                "window": 60,
                "mean": float(np.mean(valid)) if valid else None,
                "std": float(np.std(valid)) if valid else None,
            }
        vol = frame.log_return.rolling(20).std().iloc[origins].to_numpy()
        momentum = frame.close.pct_change(20).iloc[origins].to_numpy()
        result["regimes"] = {}
        for regime, regime_mask in {
            "high_volatility": vol > volatility_cut,
            "low_volatility": vol <= volatility_cut,
            "positive_past_momentum": momentum > 0,
            "nonpositive_past_momentum": momentum <= 0,
        }.items():
            if regime_mask.sum():
                result["regimes"][regime] = evaluate(
                    y[regime_mask], {k: v[regime_mask] if v is not None else None for k, v in pred.items()}, train_y
                )
        summary["models"][name] = result
        predictions.append(
            pd.DataFrame(
                {
                    "model": name,
                    "timestamp": frame.timestamp.iloc[origins].to_numpy(),
                    "target_timestamp": frame.timestamp.iloc[samples.ends[test]].to_numpy(),
                    "horizon": config.horizon,
                    "actual": y,
                    "predicted": pred.get("return"),
                    "probability": pred.get("probability"),
                    "direction": direction,
                    "dataset_version": manifest["dataset_version"],
                    "feature_version": FEATURE_VERSION,
                }
            )
        )

    for name, pred in naive_predictions(train_y, len(y)).items():
        record(name, pred)
    arima = ARIMAForecaster().fit(frame.log_return.iloc[1:train_end].to_numpy())
    start = time.perf_counter()
    arima_pred = arima.rolling_predict(frame.log_return.to_numpy(), origins, train_end - 1, config.horizon)
    record(
        "arima",
        {"return": arima_pred, "probability": None},
        {
            "order": list(arima.order),
            "order_selection": arima.selection,
            "training_seconds": arima.training_seconds,
            "latency_ms_per_sample": (time.perf_counter() - start) * 1000 / len(y),
            "state_update": "append actual daily returns through origin; coefficients frozen",
        },
    )
    for family in ("lstm", "gru"):
        for task in tasks:
            candidates = []
            for loss in ("huber",) if task == "direction" else losses:
                models, scores = [], []
                for seed in seeds:
                    c = replace(config, family=family, task=task, loss=loss, seed=seed)
                    model = create_forecaster(len(columns), c).fit(x[train], train_y, x[val], samples.y[val])
                    score = objective(samples.y[val], model.predict(x[val]))
                    scores.append(score)
                    models.append(model)
                candidate = f"{family}-{task}-{loss}"
                summary["tuning"].append(
                    {
                        "candidate": candidate,
                        "validation_objective": float(np.mean(scores)),
                        "seed_objectives": scores,
                        "selection_metric": "mse" if task != "direction" else "brier",
                    }
                )
                candidates.append((np.mean(scores), loss, models))
            _, selected_loss, selected_models = min(candidates, key=lambda c: c[0])
            summary["selection"][f"{family}-{task}"] = selected_loss
            for model in selected_models:
                name = f"{family}_{task}_seed{model.config.seed}"
                bundle = save_bundle(model, scaler, columns, manifest, summary["split"], artifacts)
                model.predict(x[test][:1])  # warmup excluded from batch latency
                start = time.perf_counter()
                pred = model.predict(x[test])
                elapsed = time.perf_counter() - start
                record(
                    name,
                    pred,
                    {**model.metadata(), "bundle": str(bundle), "latency_ms_per_sample": elapsed * 1000 / len(y)},
                )
            # Walk-forward uses the same fixed search budget, independently selected inside each fold.
            for fold in range(folds):
                end = int(validation_end * (0.75 + 0.25 * (fold + 1) / max(folds, 1)))
                fold_train_end, fold_val_end = int(end * 0.6), int(end * 0.8)
                fm = split_masks(samples, fold_train_end, fold_val_end, end)
                ft, fv, fe = fm
                fs, fr = fit_scaler(frame, columns, samples, ft, config.sequence_length)
                fx = transform(samples.x, fs)
                fold_result = {
                    "fold": fold,
                    "family": family,
                    "task": task,
                    "split": split_metadata(frame, samples, fm, fr),
                    "models": {},
                }
                fold_candidates = []
                for loss in ("huber",) if task == "direction" else losses:
                    fitted = [
                        create_forecaster(
                            len(columns), replace(config, family=family, task=task, loss=loss, seed=seed)
                        ).fit(fx[ft], samples.y[ft], fx[fv], samples.y[fv])
                        for seed in seeds
                    ]
                    score = np.mean([objective(samples.y[fv], m.predict(fx[fv])) for m in fitted])
                    fold_candidates.append((score, loss, fitted))
                _, fold_loss, fitted = min(fold_candidates, key=lambda c: c[0])
                fold_result["selected_loss"] = fold_loss
                for m in fitted:
                    fold_result["models"][str(m.config.seed)] = evaluate(
                        samples.y[fe], m.predict(fx[fe]), samples.y[ft]
                    )
                fold_result["baselines"] = {
                    n: evaluate(samples.y[fe], p, samples.y[ft])
                    for n, p in naive_predictions(samples.y[ft], int(fe.sum())).items()
                }
                fa = ARIMAForecaster().fit(frame.log_return.iloc[1:fold_train_end].to_numpy())
                fp = fa.rolling_predict(
                    frame.log_return.to_numpy(), samples.origins[fe], fold_train_end - 1, config.horizon
                )
                fold_result["baselines"]["arima"] = evaluate(
                    samples.y[fe], {"return": fp, "probability": None}, samples.y[ft]
                )
                summary["walk_forward"].append(fold_result)
    pd.concat(predictions, ignore_index=True).to_csv(root / "predictions.csv", index=False)
    write_json(root / "benchmark_summary.json", summary)
    make_reports(root)
    tracking.dataset(manifest, summary["split"], columns)
    tracking.params({"seeds": seeds, "tasks": tasks, "losses": losses, "folds": folds})
    tracking.document(summary, "benchmark.json")
    tracking.artifacts(root, "reports")
    for name, entry in summary["models"].items():
        setting = config
        metadata = None
        if entry.get("bundle"):
            import json

            metadata = json.loads((Path(entry["bundle"]) / "metadata.json").read_text(encoding="utf-8"))
            setting = Config(**metadata["model"]["config"])
        with tracking_run(
            setting, "research_benchmark_model", parent=tracking, name=name, family=None if metadata else name
        ) as child:
            child.dataset(manifest, summary["split"], columns)
            child.metrics(entry["metrics"], "test")
            child.document(entry, "evaluation.json")
            if metadata:
                child.history(metadata["model"]["history"])
                child.tags({"actual_device": metadata["model"]["device"]})
                child.artifacts(entry["bundle"], "model")
    summary["mlflow_run_id"] = tracking.run_id
    return summary
