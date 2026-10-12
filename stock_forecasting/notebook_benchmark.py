"""Multi-stock notebook experiment. Results stay in memory; export Markdown and PNG only."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
import torch

from .backtesting import backtest
from .benchmark import split_metadata
from .data import features, fit_scaler, normalize, read_csv, sequences, split_masks, targets, transform
from .evaluation import evaluate
from .storage import snapshot_identity
from .tracking import tracking_run
from .models import ARIMAForecaster, create_forecaster, naive_predictions


def prepare_stock(path, config, validation_start="2022-01-01", test_start="2024-01-01"):
    """All stocks share date boundaries; training/scaling use no validation or test rows."""
    frame, quality = normalize(read_csv(path), config)
    quality["source_file_sha256"] = sha256(Path(path).read_bytes()).hexdigest()
    manifest, _ = snapshot_identity(frame, quality)
    frame, columns = features(frame, config.feature_groups)
    frame = targets(frame, config.horizon)
    samples = sequences(frame, columns, config.sequence_length, config.horizon)
    train_end = int(frame.timestamp.searchsorted(pd.Timestamp(validation_start)))
    val_end = int(frame.timestamp.searchsorted(pd.Timestamp(test_start)))
    masks = split_masks(samples, train_end, val_end)
    scaler, rows = fit_scaler(frame, columns, samples, masks[0], config.sequence_length)
    return {
        "config": config,
        "frame": frame,
        "columns": columns,
        "samples": samples,
        "masks": masks,
        "x": transform(samples.x, scaler),
        "quality": quality,
        "manifest": manifest,
        "split": split_metadata(frame, samples, masks, rows),
        "train_end": train_end,
        "source_sha256": sha256(Path(path).read_bytes()).hexdigest(),
        "validation_start": validation_start,
        "test_start": test_start,
    }


def benchmark_stock(data, seeds=(7, 19, 43), mlflow_uri=None, mlflow_experiment=None):
    with tracking_run(data["config"], "benchmark", mlflow_uri, mlflow_experiment) as tracking:
        tracking.dataset(data["manifest"], data["split"], data["columns"])
        tracking.params(
            {"seeds": seeds, "validation_start": data["validation_start"], "test_start": data["test_start"]}
        )
        tracking.tags({"report_exported": "false"})
        result = _benchmark_stock(data, seeds, tracking)
        result["tracking"] = tracking
        tracking.document(result["budgets"].to_dict(orient="records"), "budgets.json")
        return result


def _benchmark_stock(data, seeds, tracking):
    """Fit each family/task/seed once. Evaluate validation and test separately.

    Hyperparameters are fixed in advance; validation picks epochs. Test rankings
    are descriptive, never used to pick a seed or tune the trading threshold.
    """
    c, frame, samples = data["config"], data["frame"], data["samples"]
    if c.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run; CPU fallback is disabled")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    train, val, test = data["masks"]
    x = data["x"]
    rows, displayed, histories, budgets = [], {}, {}, []

    def record(name, phase, mask, prediction, seed=None, seconds=None, latency=None, run=None):
        current_run = run or tracking
        scores = evaluate(samples.y[mask], prediction, samples.y[train])
        prefix = phase if run is not None else f"{name}.{phase}"
        current_run.metrics(scores, prefix)
        current_run.metrics({"training_seconds": seconds, "prediction_ms_per_sample": latency}, "runtime")
        current_run.document(scores, f"{name}-{phase}-metrics.json")
        rows.append(
            {
                "symbol": c.symbol,
                "model": name,
                "phase": phase,
                "seed": seed,
                "n": int(mask.sum()),
                "rmse": scores.get("regression", {}).get("rmse"),
                "mae": scores.get("regression", {}).get("mae"),
                "accuracy": scores.get("direction", {}).get("accuracy"),
                "brier": scores.get("probability", {}).get("brier"),
                "auc": scores.get("probability", {}).get("roc_auc"),
                "training_seconds": seconds,
                "prediction_ms_per_sample": latency,
            }
        )
        if phase == "test" and (seed is None or seed == seeds[0]):
            displayed[name] = {"prediction": prediction, "metrics": scores}

    for name in naive_predictions(samples.y[train], int(test.sum())):
        with tracking_run(c, "benchmark_baseline", parent=tracking, name=f"{c.symbol}-{name}", family=name) as baseline:
            baseline.dataset(data["manifest"], data["split"], data["columns"])
            for phase, mask in (("validation", val), ("test", test)):
                pred = naive_predictions(samples.y[train], int(mask.sum()))[name]
                record(name, phase, mask, pred, run=baseline)
    print(f"{c.symbol}: fitting ARIMA on CPU", flush=True)
    with tracking_run(c, "benchmark_baseline", parent=tracking, name=f"{c.symbol}-ARIMA", family="arima") as baseline:
        baseline.dataset(data["manifest"], data["split"], data["columns"])
        arima = ARIMAForecaster().fit(frame.log_return.iloc[1 : data["train_end"]].to_numpy())
        baseline.params({"arima_order": arima.order})
        baseline.tags({"actual_device": "cpu"})
        baseline.document(arima.selection, "arima_selection.json")
        for phase, mask in (("validation", val), ("test", test)):
            start = time.perf_counter()
            forecast = arima.rolling_predict(
                frame.log_return.to_numpy(), samples.origins[mask], data["train_end"] - 1, c.horizon
            )
            record(
                "ARIMA",
                phase,
                mask,
                {"return": forecast, "probability": None},
                seconds=arima.training_seconds,
                latency=(time.perf_counter() - start) * 1000 / mask.sum(),
                run=baseline,
            )
    for task in ("return", "direction"):
        for family in ("lstm", "gru"):
            name = f"{family.upper()}_{task}"
            for seed in seeds:
                setting = replace(c, family=family, task=task, seed=seed)
                with tracking_run(
                    setting, "benchmark_model", parent=tracking, name=f"{c.symbol}-{name}-seed{seed}"
                ) as current_run:
                    current_run.dataset(data["manifest"], data["split"], data["columns"])
                    model = create_forecaster(len(data["columns"]), setting)
                    model.fit(x[train], samples.y[train], x[val], samples.y[val])
                    meta = model.metadata()
                    current_run.tags({"actual_device": meta["device"]})
                    current_run.document(meta, "model.json")
                    histories[(name, seed)] = model.history
                    best_epoch = min(model.history, key=lambda row: row["validation_loss"])["epoch"]
                    budgets.append(
                        {
                            "model": name,
                            "seed": seed,
                            "epochs_run": len(model.history),
                            "best_epoch": best_epoch,
                            "device": meta["device"],
                            "seconds": meta["training_seconds"],
                        }
                    )
                    for phase, mask in (("validation", val), ("test", test)):
                        model.predict(x[mask][:1])  # Warmup is not included in measured latency.
                        if c.device == "cuda":
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        prediction = model.predict(x[mask])
                        if c.device == "cuda":
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        record(
                            name,
                            phase,
                            mask,
                            prediction,
                            seed,
                            meta["training_seconds"],
                            elapsed * 1000 / mask.sum(),
                            current_run,
                        )
                    print(
                        f"{c.symbol}: {name}, seed {seed}, {len(model.history)}/{c.epochs} epochs, "
                        f"best {best_epoch}, {meta['training_seconds']:.1f}s on {meta['device']}",
                        flush=True,
                    )
                    del model
                    current_run.history(histories[(name, seed)])
    strategies = {}
    for name, entry in displayed.items():
        pred = entry["prediction"]
        signal = pred.get("return")
        if signal is None:
            signal = (
                pred["probability"] - 0.5
                if pred.get("probability") is not None
                else pred["direction"].astype(float) - 0.5
            )
        try:
            metrics, curve, _ = backtest(frame, samples.origins[test], signal, c.horizon, c.threshold, c.cost_bps)
            strategies[name] = {"metrics": metrics, "curve": curve}
        except ValueError as exc:
            strategies[name] = {"unavailable": str(exc)}
    return {
        **data,
        "scores": pd.DataFrame(rows),
        "predictions": displayed,
        "histories": histories,
        "budgets": pd.DataFrame(budgets),
        "strategies": strategies,
        "seeds": tuple(seeds),
        "arima_order": arima.order,
        "arima_selection": arima.selection,
    }


def comparison_tables(results):
    """Equal weight per stock; seed averages are not ensemble predictions."""
    scores = pd.concat([r["scores"] for r in results.values()], ignore_index=True)
    per_stock = (
        scores.groupby(["symbol", "phase", "model"], sort=False)
        .agg(
            rmse=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae=("mae", "mean"),
            accuracy=("accuracy", "mean"),
            brier=("brier", "mean"),
            auc=("auc", "mean"),
            training_seconds=("training_seconds", "mean"),
            prediction_ms_per_sample=("prediction_ms_per_sample", "mean"),
        )
        .reset_index()
    )
    baseline = per_stock[per_stock.model == "historical_mean"].set_index(["symbol", "phase"]).rmse
    probability = per_stock[per_stock.model == "historical_probability"].set_index(["symbol", "phase"]).brier
    per_stock["relative_rmse"] = [
        row.rmse / baseline[row.symbol, row.phase] if baseline[row.symbol, row.phase] > 0 else np.nan
        for row in per_stock.itertuples()
    ]
    per_stock["relative_brier"] = [
        row.brier / probability[row.symbol, row.phase] if probability[row.symbol, row.phase] > 0 else np.nan
        for row in per_stock.itertuples()
    ]
    overall = (
        per_stock.groupby(["phase", "model"], sort=False)[
            ["relative_rmse", "relative_brier", "mae", "accuracy", "training_seconds", "prediction_ms_per_sample"]
        ]
        .mean()
        .reset_index()
    )
    return per_stock, overall


def model_choice_text(per_stock, overall):
    """Choose using validation only; describe test performance without reselecting."""
    validation = overall[(overall.phase == "validation") & overall.relative_rmse.notna()]
    winner = validation.sort_values("relative_rmse").iloc[0]
    test = overall[(overall.phase == "test") & overall.relative_rmse.notna()]
    test_winner = test.sort_values("relative_rmse").iloc[0]
    candidate_test = test[test.model == winner.model].iloc[0]
    stock_scores = per_stock[(per_stock.phase == "validation") & per_stock.relative_rmse.notna()]
    candidate = stock_scores[stock_scores.model == winner.model]
    wins = stock_scores.loc[stock_scores.groupby("symbol").relative_rmse.idxmin()]
    count = len(candidate)
    return (
        f"**Provisional return-model choice: {winner.model}.** The selection rule is the lowest "
        "equal-stock mean validation relative RMSE, after averaging seeds within each stock. "
        "Direction models are evaluated separately because probability prediction is a different task.\n\n"
        f"The candidate scores **{winner.relative_rmse:.6f}** on validation (historical-mean baseline = 1), "
        f"an error reduction of **{100 * (1 - winner.relative_rmse):.3f}%** relative to that baseline. "
        f"It beats the baseline on **{int((candidate.relative_rmse < 1).sum())}/{count}** stocks and has "
        f"the lowest validation return error on **{int((wins.model == winner.model).sum())}/{count}** stocks.\n\n"
        f"On test, this candidate scores **{candidate_test.relative_rmse:.6f}**; "
        f"the lowest observed test error belongs to **{test_winner.model}** "
        f"(**{test_winner.relative_rmse:.6f}**). Test rankings describe generalization and do not "
        "change the validation-based choice. A score above 1 means worse error than the baseline.\n\n"
        "This supports a candidate for further validation, not an unconditional production winner. "
        "Small average differences and inconsistent stock-level results do not establish a robust advantage; "
        "no statistical significance claim is made. Keep ARIMA and the simple baselines as challengers. "
        "Before deployment, freeze the candidate settings and evaluate newly collected, unseen observations "
        "against those challengers with a predefined acceptance rule. If the candidate does not improve "
        "reliably, keep the stronger baseline rather than selecting LSTM because it was preferred."
    )


def markdown_table(table):
    """Small readable tables without an extra tabulate dependency."""

    def text(value):
        if pd.isna(value):
            return "—"
        return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)

    return "\n".join(
        [
            "| " + " | ".join(table.columns) + " |",
            "|" + "|".join(["---"] * len(table.columns)) + "|",
            *["| " + " | ".join(text(v) for v in row) + " |" for row in table.itertuples(index=False, name=None)],
        ]
    )


def stock_figures(result, directory):
    """Six grouped figures per stock, with a fixed first seed for prediction plots."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    c, frame, samples = result["config"], result["frame"], result["samples"]
    train, val, test = result["masks"]
    dates = frame.timestamp.iloc[samples.origins[test]]
    y = samples.y[test]
    predictions = result["predictions"]
    return_names = ("ARIMA", "LSTM_return", "GRU_return")
    direction_names = ("LSTM_direction", "GRU_direction")

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(frame.timestamp, frame.close)
    axes[0].set(ylabel="Provider close", title=f"{c.symbol}: all available history from 2010")
    axes[1].plot(frame.timestamp, frame.log_return, linewidth=0.6)
    axes[1].set_ylabel("Daily log return")
    for ax in axes:
        for name, mask in (("Validation", val), ("Test", test)):
            ax.axvline(frame.timestamp.iloc[samples.origins[mask][0]], linestyle="--", label=name)
        ax.legend()
    save(fig, "01_history.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, name in zip(axes.flat, (*return_names[1:], *direction_names)):
        for seed in result["seeds"]:
            history = pd.DataFrame(result["histories"][name, seed])
            (line,) = ax.plot(history.epoch, history.train_loss, label=f"Train {seed}", alpha=0.8)
            ax.plot(history.epoch, history.validation_loss, linestyle="--", color=line.get_color(), label=f"Val {seed}")
        ax.set(title=name, xlabel="Epoch", ylabel="Loss")
        ax.legend(fontsize=7, ncol=2)
    save(fig, "02_training.png")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    axes[0, 0].plot(dates, y, color="black", alpha=0.4, label="Actual")
    origin_close = frame.close.iloc[samples.origins[test]].to_numpy()
    axes[0, 1].plot(dates, origin_close * np.exp(y), color="black", label="Actual target close")
    for name in return_names:
        pred = predictions[name]["prediction"]["return"]
        axes[0, 0].plot(dates, pred, label=name, alpha=0.7)
        axes[0, 1].plot(dates, origin_close * np.exp(pred), label=name, alpha=0.7)
        axes[1, 0].hist(pred - y, bins=50, alpha=0.4, label=name)
        axes[1, 1].plot(dates, pd.Series(abs(pred - y)).rolling(60).mean(), label=name)
    for ax, title in zip(
        axes.flat,
        (
            "Actual / predicted log returns",
            "Target prices by forecast origin (not recursive)",
            "Prediction minus actual",
            "Rolling 60-origin MAE",
        ),
    ):
        ax.set_title(title)
        ax.legend(fontsize=8)
    save(fig, "03_forecast.png")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, name in zip(axes.flat, (*return_names, *direction_names, "majority_direction")):
        cm = np.asarray(predictions[name]["metrics"]["direction"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", color="white" if value > cm.max() / 2 else "black")
        ax.set(
            title=name,
            xlabel="Predicted",
            ylabel="Actual",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Nonpositive", "Positive"],
            yticklabels=["Nonpositive", "Positive"],
        )
    save(fig, "04_confusion.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for name in direction_names:
        p = predictions[name]["prediction"]["probability"]
        if len(np.unique(y > 0)) == 2:
            fpr, tpr, _ = roc_curve(y > 0, p)
            precision, recall, _ = precision_recall_curve(y > 0, p)
            axes[0, 0].plot(fpr, tpr, label=name)
            axes[0, 1].plot(recall, precision, label=name)
        bins = predictions[name]["metrics"]["probability"]["calibration"]
        axes[1, 0].plot([b["mean_probability"] for b in bins], [b["positive_fraction"] for b in bins], "o-", label=name)
        axes[1, 1].hist(p, bins=np.linspace(0, 1, 21), alpha=0.5, label=name)
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0, 1].axhline(np.mean(y > 0), linestyle="--", color="black", label="Positive fraction")
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    for ax, title, xlabel, ylabel in zip(
        axes.flat,
        ("ROC", "Precision-recall", "Reliability (uncalibrated)", "Probability distribution"),
        ("False-positive rate", "Recall", "Predicted probability", "Probability up"),
        ("True-positive rate", "Precision", "Observed positive fraction", "Count"),
    ):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=8)
    save(fig, "05_probability.png")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    benchmark_drawn = False
    for name in (*return_names, *direction_names):
        strategy = result["strategies"][name]
        if "curve" not in strategy:
            continue
        curve = strategy["curve"]
        axes[0].plot(curve.timestamp, curve.equity, label=name)
        axes[1].plot(curve.timestamp, curve.drawdown, label=name)
        if not benchmark_drawn:
            axes[0].plot(curve.timestamp, curve.buy_hold, "k--", label="Gross buy & hold")
            benchmark_drawn = True
    axes[0].set(title=f"Next-open strategy, {c.cost_bps:g} bps per side", ylabel="Equity (start = 1)")
    axes[1].set_ylabel("Drawdown")
    if benchmark_drawn:
        axes[0].legend(fontsize=8, ncol=2)
    else:
        axes[0].text(0.1, 0.5, "Unavailable: missing liquidity/fill evidence", transform=axes[0].transAxes)
    save(fig, "06_strategy.png")


def export_benchmark(results, output):
    try:
        return _export_benchmark(results, output)
    except BaseException:
        for result in results.values():
            tracking = result.get("tracking")
            if tracking and tracking.client:
                tracking.tags({"report_exported": "failed"})
                tracking.client.set_terminated(tracking.run_id, "FAILED")
        raise


def _export_benchmark(results, output):
    """Write one Markdown report and grouped PNG figures; no CSV/JSON/model files."""
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    per_stock, overall = comparison_tables(results)
    first = next(iter(results.values()))
    config = first["config"]
    validation = overall[(overall.phase == "validation") & overall.relative_rmse.notna()].sort_values("relative_rmse")
    neural = validation[validation.model.isin(["LSTM_return", "GRU_return"])]
    preferred = neural.iloc[0].model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, phase in zip(axes, ("validation", "test")):
        table = per_stock[(per_stock.phase == phase) & per_stock.model.isin(["ARIMA", "LSTM_return", "GRU_return"])]
        table.pivot(index="symbol", columns="model", values="relative_rmse").plot.bar(ax=ax)
        ax.axhline(1, color="black", linestyle="--", label="Training-mean baseline")
        ax.set(title=phase.capitalize(), ylabel="RMSE / baseline RMSE (lower better)", xlabel="Stock")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figures/00_stock_comparison.png", dpi=130)
    plt.close(fig)
    lines = [
        f"# {len(results)}-stock forecasting benchmark",
        "",
        "## Scope and decision",
        "",
        f"Stocks: **{', '.join(results)}**. Neural training and prediction device: **{config.device}**; "
        f"GPU: **{torch.cuda.get_device_name(0) if config.device == 'cuda' else 'not used'}**. ARIMA runs on CPU.",
        "",
        f"The lower equal-stock mean validation relative RMSE among neural return models is **{preferred}**. "
        f"The lowest validation score across return models/baselines is **{validation.iloc[0].model}**. "
        "This is a retrospective comparison, not deployment approval or a profitability claim. Test results do not choose the winner.",
        "",
        "## Model-choice rationale",
        "",
        model_choice_text(per_stock, overall),
        "",
        "## Fixed protocol",
        "",
        f"- Start requested: {config.start}; as-of cutoff: {config.as_of}; each stock uses all available completed rows within that range.",
        f"- Horizon: {config.horizon} observed bar(s). Unverified calendar means these are not guaranteed exchange sessions.",
        f"- Maximum epochs: {config.epochs}; patience: {config.patience}; seeds: {list(first['seeds'])}.",
        f"- Window: {config.sequence_length}; hidden units: {config.hidden_size}; layers: {config.layers}; dropout: {config.dropout}.",
        f"- Batch size: {config.batch_size}; learning rate: {config.learning_rate}; weight decay: {config.weight_decay}.",
        f"- Return objective: {config.loss} on training-standardized targets; direction objective: binary cross entropy.",
        "- Same public LSTMModel / GRUModel and trainer as routine training. Benchmark overrides history, epoch budget and device; it never refits on test or promotes models.",
        f"- Training before {first['validation_start']}; validation until {first['test_start']} (exclusive); "
        "test thereafter. Future labels crossing boundaries are purged. Scalers use training inputs only.",
        "- Both families run each seed for return prediction and separately for direction probability. Validation restores the best epoch; no test-based hyperparameter, seed or trading-threshold selection.",
        "- Model tables average metrics across seeds (not ensemble predictions). Stocks receive equal weight. Relative RMSE uses each stock's historical-mean baseline; relative Brier uses its training-frequency baseline. Values below 1 beat that baseline.",
        f"- Forecast, confusion, probability and strategy figures use seed {first['seeds'][0]} fixed in advance. Learning curves include all seeds.",
        "- Neural latency includes input transfer and CPU result transfer, with warmup excluded and CUDA synchronization. ARIMA latency includes sequential state updates; these workloads differ. GPU speedup over CPU was not measured.",
        "- Stocks were selected for history coverage and valid local OHLCV, not benchmark performance. Actual coverage appears in each stock section. Invalid stocks were not silently repaired.",
        "- Existing historical data and some test periods have been inspected in prior work; this is not a newly unseen prospective test. Unverified corporate actions, data revisions and missing-session evidence limit conclusions.",
        "",
        "## Overall results",
        "",
        markdown_table(overall),
        "",
        "![Across-stock comparison](figures/00_stock_comparison.png)",
        "",
        "## Reading the metrics",
        "",
        "RMSE and MAE measure errors in decimal log returns (lower is better). RMSE emphasizes large errors. "
        "Accuracy is the fraction of correct positive/nonpositive signs; compare it with the majority baseline. "
        "Brier measures squared probability error (lower is better); ROC-AUC measures ranking (higher is better). "
        "Probabilities are uncalibrated sigmoid outputs from trained direction heads, not signs disguised as confidence. "
        "Unavailable metrics appear as —. Seed standard deviation describes training variability, not a confidence interval.",
        "",
        "Strategies form signals after close, enter at the next open, and hold a nonoverlapping fixed horizon. "
        f"Costs are {config.cost_bps:g} bps on each side. Probability signals use 0.5. Buy-and-hold is gross. "
        "Zero-volume execution bars make this simple simulation unavailable. Charts are research diagnostics, not deployable trading evidence.",
    ]
    for symbol, result in results.items():
        print(f"Exporting grouped figures: {symbol}", flush=True)
        stock_figures(result, output / "figures" / symbol)
        q = result["quality"]
        lines += [
            "",
            f"## {symbol}",
            "",
            f"{q['rows']} rows: {q['start']} to {q['end']}. "
            f"ARIMA order: {result['arima_order']}. Input SHA-256: `{result['source_sha256']}`.",
            "",
            markdown_table(
                pd.DataFrame(
                    [
                        {
                            "split": phase,
                            "n": result["split"][f"{phase}_n"],
                            "first_origin": result["split"][f"{phase}_start"],
                            "last_origin": result["split"][f"{phase}_end"],
                            "last_label": result["split"][f"{phase}_last_label"],
                        }
                        for phase in ("train", "validation", "test")
                    ]
                )
            ),
            "",
            markdown_table(per_stock[per_stock.symbol == symbol].drop(columns="symbol")),
            "",
            "### Epochs actually used",
            "",
            markdown_table(result["budgets"]),
            "",
        ]
        strategy_rows = []
        for name, strategy in result["strategies"].items():
            m = strategy.get("metrics", {})
            strategy_rows.append(
                {
                    "model": name,
                    "net_return": m.get("net_cumulative_return"),
                    "max_drawdown": m.get("max_drawdown"),
                    "trades": m.get("number_of_trades"),
                    "status": strategy.get("unavailable", "available"),
                }
            )
        lines += ["### Strategy diagnostics", "", markdown_table(pd.DataFrame(strategy_rows)), ""]
        for filename, title in (
            ("01_history", "History and splits"),
            ("02_training", "Training and validation curves"),
            ("03_forecast", "Forecasts and errors"),
            ("04_confusion", "Confusion matrices"),
            ("05_probability", "Probability diagnostics"),
            ("06_strategy", "Strategy and drawdown"),
        ):
            lines += [f"### {title}", "", f"![{symbol}: {title}](figures/{symbol}/{filename}.png)", ""]
    lines += [
        "## Files and next decision",
        "",
        "This report and the figures are the only exported results. "
        "Displayed tables are saved in notebook outputs; full predictions exist only in the live kernel and require rerunning after restart. "
        "No deployment model or data snapshot is written. "
        "Review validation consistency across stocks before accepting a model family. This report does not establish that LSTM is best, "
        "nor guarantee that any learned model improves on simple baselines. Future monitoring and release gates remain separate work.",
        "",
    ]
    report = output / "benchmark_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    for result in results.values():
        tracking = result.get("tracking")
        if tracking:
            tracking.tags({"report_exported": "false"})
            tracking.artifacts(report, "reports")
            # Keep every relative figure link in the uploaded report valid.
            tracking.artifacts(output / "figures", "reports/figures")
            tracking.tags({"report_exported": "true"})
            if tracking.client:
                tracking.client.set_terminated(tracking.run_id, "FINISHED")
    return report
