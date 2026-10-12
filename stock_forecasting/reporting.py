"""Rebuild portfolio figures and Markdown from persisted experiment outputs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_curve
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .evaluation import correlation


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def make_reports(root):
    root = Path(root)
    summary = json.loads((root / "benchmark_summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(root / "history.csv", parse_dates=["timestamp"])
    predictions = pd.read_csv(root / "predictions.csv", parse_dates=["timestamp", "target_timestamp"])
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figdir / f"{name}.png", dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(frame.timestamp, frame.close)
    axes[0].set_ylabel("Provider close")
    returns = np.log(frame.close).diff()
    axes[1].plot(frame.timestamp, returns)
    axes[1].set_ylabel("Log return (decimal)")
    axes[2].plot(frame.timestamp, returns.rolling(20).mean(), label="20-bar mean")
    axes[2].plot(frame.timestamp, returns.rolling(20).std(), label="20-bar std")
    axes[2].plot(frame.timestamp, returns.rolling(20).var(), label="20-bar variance")
    axes[2].legend()
    for ax in axes:
        for key in ("validation_start", "test_start"):
            ax.axvline(pd.Timestamp(summary["split"][key]), color="red", linestyle="--")
    save(fig, "history_and_splits")
    train = frame[frame.timestamp < pd.Timestamp(summary["split"]["validation_start"])]
    tr = np.log(train.close).diff().dropna()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes[0, 0].hist(train.close, bins=40)
    axes[0, 0].set_title("Training price distribution")
    axes[0, 1].hist(tr, bins=50)
    axes[0, 1].set_title("Training return distribution")
    stats.probplot(tr, plot=axes[0, 2])
    axes[0, 2].set_title("Normal Q-Q")
    plot_acf(tr, lags=min(30, len(tr) // 3), ax=axes[1, 0])
    plot_pacf(tr, lags=min(30, len(tr) // 3), ax=axes[1, 1], method="ywm")
    plot_acf(tr**2, lags=min(30, len(tr) // 3), ax=axes[1, 2])
    axes[1, 2].set_title("Squared return ACF")
    save(fig, "training_diagnostics")
    rows = []
    for name, result in summary["models"].items():
        metric = result["metrics"]
        rows.append(
            {
                "model": name,
                "rmse": metric.get("regression", {}).get("rmse"),
                "direction_accuracy": metric.get("direction", {}).get("accuracy"),
                "brier": metric.get("probability", {}).get("brier"),
                "net_return": result.get("backtest", {}).get("net_cumulative_return"),
                "training_seconds": result.get("training_seconds"),
                "parameters": result.get("parameters"),
                "latency_ms_per_sample": result.get("latency_ms_per_sample"),
            }
        )
        part = predictions[predictions.model == name]
        actual, forecast = part.actual.to_numpy(), part.predicted.to_numpy()
        if np.isfinite(forecast).all():
            error = forecast - actual
            fig, axes = plt.subplots(3, 2, figsize=(13, 9))
            axes[0, 0].plot(part.timestamp, actual, label="actual", alpha=0.6)
            axes[0, 0].plot(part.timestamp, forecast, label="predicted")
            axes[0, 0].legend()
            axes[0, 1].hist(error, bins=40)
            axes[0, 1].set_title("Prediction minus actual")
            axes[1, 0].plot(part.timestamp, error)
            axes[1, 0].set_title("Error over time")
            axes[1, 1].plot(part.timestamp, pd.Series(abs(error)).rolling(60).mean())
            axes[1, 1].set_title("Rolling 60-origin MAE")
            rolling_ic = [
                correlation(actual[max(0, i - 59) : i + 1], forecast[max(0, i - 59) : i + 1], True) if i >= 59 else None
                for i in range(len(part))
            ]
            axes[2, 0].plot(part.timestamp, rolling_ic)
            axes[2, 0].set_title("Rolling time-series rank IC")
            axes[2, 1].plot(part.timestamp, pd.Series((actual > 0) == (forecast > 0)).rolling(60).mean())
            axes[2, 1].set_title("Rolling directional accuracy")
            fig.suptitle(name + " — decimal log returns")
            save(fig, f"{name}_forecast")
        if "direction" in metric:
            fig, ax = plt.subplots(figsize=(4, 4))
            cm = np.asarray(metric["direction"]["confusion_matrix"])
            ax.imshow(cm, cmap="Blues")
            for (i, j), value in np.ndenumerate(cm):
                ax.text(j, i, str(value), ha="center")
            ax.set(xlabel="Predicted (0 nonpositive, 1 positive)", ylabel="Actual", title=name)
            save(fig, f"{name}_confusion")
        if "probability" in metric:
            probability = part.probability.to_numpy()
            fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
            if len(np.unique(actual > 0)) == 2:
                fpr, tpr, _ = roc_curve(actual > 0, probability)
                precision, recall, _ = precision_recall_curve(actual > 0, probability)
                axes[0].plot(fpr, tpr)
                axes[0].set_title("ROC")
                axes[1].plot(recall, precision)
                axes[1].set_title("Precision-recall")
            bins = metric["probability"]["calibration"]
            axes[2].plot([b["mean_probability"] for b in bins], [b["positive_fraction"] for b in bins], "o-")
            axes[2].plot([0, 1], [0, 1], "--")
            axes[2].set_title("Reliability (uncalibrated)")
            for outcome in (False, True):
                axes[3].hist(probability[(actual > 0) == outcome], alpha=0.5, label=str(outcome))
            axes[3].legend()
            axes[3].set_title("Probability by outcome")
            save(fig, f"{name}_probability")
        curve_path = root / "curves" / f"{name}.csv"
        if curve_path.exists():
            curve = pd.read_csv(curve_path, parse_dates=["timestamp"])
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            axes[0].plot(curve.timestamp, curve.equity, label="net strategy")
            axes[0].plot(curve.timestamp, curve.buy_hold, label="gross buy/hold")
            axes[0].legend()
            axes[1].fill_between(curve.timestamp, curve.drawdown, 0)
            axes[1].set_ylabel("Drawdown")
            save(fig, f"{name}_strategy")
        if result.get("history"):
            history = pd.DataFrame(result["history"])
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(history.epoch, history.train_loss, label="train")
            ax.plot(history.epoch, history.validation_loss, label="validation")
            ax.legend()
            save(fig, f"{name}_training")
    table = pd.DataFrame(rows)
    table.to_csv(root / "benchmark_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key in zip(axes, ["rmse", "direction_accuracy", "brier"]):
        clean = table.dropna(subset=[key])
        ax.barh(clean.model, clean[key])
        ax.set_title(key)
    save(fig, "benchmark_comparison")
    tuning = pd.DataFrame(summary.get("tuning", []))
    if not tuning.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(tuning.candidate, tuning.validation_objective)
        ax.set_title("Validation-only objective (compare within task)")
        save(fig, "hyperparameter_comparison")
    lines = [
        "# Reproducible benchmark report",
        "",
        f"Dataset: `{summary['dataset']['dataset_version']}`; {summary['dataset']['symbol']}, "
        f"{summary['dataset']['start']} to {summary['dataset']['end']}.",
        "",
        f"Horizon: {summary['config']['horizon']}; basis: **{summary['dataset']['horizon_basis']}**; "
        f"adjustments: **{summary['dataset']['adjustment']}**.",
        "",
        "Validation selects loss settings; seeds are replicated, not cherry-picked. Final holdout is scored once per selected model/seed. "
        "Walk-forward development results are separate. No test-based trading thresholds. See benchmark_summary.json for exact split dates and paired statistics.",
        "",
        "| Model | RMSE | Direction accuracy | Brier | Net strategy return |",
        "|---|---:|---:|---:|---:|",
    ]

    def display(v):
        return "—" if v is None or pd.isna(v) else f"{v:.6f}"

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [row["model"]] + [display(row[k]) for k in ("rmse", "direction_accuracy", "brier", "net_return")]
            )
            + " |"
        )
    lines += [
        "",
        "Returns are decimal units. The zero-return baseline also implies previous-close price persistence. "
        "Classification scores are uncalibrated sigmoid probabilities; regression sign is never a probability.",
        "",
        "Economic simulation: after-close signal, next-open execution, fixed-horizon nonoverlapping holdings. "
        "Costs and borrow assumptions are configurable. Raw/unverified corporate actions and sparse sessions prevent investable performance claims. "
        "No causal inference, profitability promise or cross-sectional IC claim follows from this single-asset experiment.",
        "",
        "![Benchmark comparison](figures/benchmark_comparison.png)",
        "",
        "![History](figures/history_and_splits.png)",
        "",
        "Training-only ADF/KPSS/ARCH/Ljung–Box results are in statistical_analysis.json. "
        "ADF rejection does not establish full stationarity. Multiple comparisons and retrospective data revisions remain limitations.",
    ]
    (root / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return table
