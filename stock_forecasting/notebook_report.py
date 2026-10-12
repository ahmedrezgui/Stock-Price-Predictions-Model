"""Export notebook results as one readable Markdown report and four figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .evaluation import evaluate


def export_report(output, config, frame, columns, samples, masks, models, predictions, quality, arima_order):
    """Use already fitted notebook models; do not train or save model/data artifacts."""
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    train, validation, test = masks
    actual = samples.y[test]
    dates = frame.timestamp.iloc[samples.origins[test]]
    rows = []
    for name, prediction in predictions.items():
        metrics = evaluate(actual, prediction, samples.y[train])
        regression = metrics.get("regression", {})
        rows.append(
            {
                "Model": name,
                "RMSE": regression.get("rmse"),
                "MAE": regression.get("mae"),
                "Direction accuracy": metrics.get("direction", {}).get("accuracy"),
                "Brier": metrics.get("probability", {}).get("brier"),
            }
        )
    table = pd.DataFrame(rows)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame.timestamp, frame.close, label="Observed close")
    for label, mask in (("Validation starts", validation), ("Test starts", test)):
        ax.axvline(frame.timestamp.iloc[samples.origins[mask][0]], linestyle="--", label=label)
    ax.set(ylabel="Provider price", title=f"{config.symbol}: history and chronological splits")
    ax.legend()
    save(fig, "01_data_and_splits.png")

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 3), squeeze=False)
    for ax, (name, model) in zip(axes[0], models.items()):
        history = pd.DataFrame(model.history)
        ax.plot(history.epoch, history.train_loss, label="Training")
        ax.plot(history.epoch, history.validation_loss, label="Validation")
        ax.set(title=name, xlabel="Epoch", ylabel="Training objective")
        ax.legend()
    save(fig, "02_learning_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("RMSE", "Direction accuracy")):
        values = table.dropna(subset=[metric])
        ax.barh(values.Model, values[metric])
        ax.set_title(metric + (" — lower is better" if metric == "RMSE" else " — higher is better"))
    save(fig, "03_model_comparison.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, actual, label="Actual", color="black", alpha=0.5)
    for name in (*models, "ARIMA"):
        forecast = predictions[name].get("return")
        if forecast is not None:
            ax.plot(dates, forecast, label=name, alpha=0.8)
    ax.set(ylabel="Log return (decimal)", title="Predictions on the same test observations")
    ax.legend()
    save(fig, "04_test_predictions.png")

    def number(value):
        return "—" if pd.isna(value) else f"{value:.6f}"

    lines = [
        f"# {config.symbol} benchmarking report",
        "",
        "## What this run answers",
        "",
        "This notebook compares LSTM, GRU, ARIMA and simple baselines on the same stock and test origins. "
        "It is a single-stock, single-seed experiment, not a final model-selection decision across stocks. "
        "Training is performed locally; no models are deployed or saved by this export.",
        "",
        "## Data and experiment settings",
        "",
        f"- Observations: {quality['start']} to {quality['end']}; {len(frame)} rows.",
        f"- Horizon: {config.horizon}; basis: `{quality['horizon_basis']}`.",
        f"- Price adjustment status: `{config.adjustment}`.",
        f"- Seed: {config.seed}; maximum epochs: {config.epochs}; early-stopping patience: {config.patience}.",
        f"- Task: `{config.task}`; loss: `{config.loss}`; sequence length: {config.sequence_length}.",
        f"- Recurrent hidden size: {config.hidden_size}; layers: {config.layers}; dropout: {config.dropout}.",
        f"- Learning rate: {config.learning_rate}; batch size: {config.batch_size}; weight decay: {config.weight_decay}.",
        f"- ARIMA order selected using training AIC: `{tuple(arima_order)}`.",
        f"- Input columns: {', '.join(columns)}.",
        "",
        "| Split | Samples | First forecast origin | Last forecast origin | Last label date |",
        "|---|---:|---|---|---|",
    ]
    for name, mask in (("Training", train), ("Validation", validation), ("Test", test)):
        origins, ends = samples.origins[mask], samples.ends[mask]
        lines.append(
            f"| {name} | {int(mask.sum())} | {frame.timestamp.iloc[origins[0]].date()} | "
            f"{frame.timestamp.iloc[origins[-1]].date()} | {frame.timestamp.iloc[ends[-1]].date()} |"
        )
    lines += [
        "",
        "Labels crossing split boundaries are purged. The scaler uses training inputs only. "
        "Early stopping restores the best validation checkpoint. ARIMA coefficients remain frozen while "
        "realized observations are appended through each forecast origin.",
        "",
        "![Data and splits](figures/01_data_and_splits.png)",
        "",
        "## Learning curves",
        "",
        "Training loss measures fit to training samples; validation loss tracks generalization during fitting. "
        "A widening gap can indicate overfitting. The default three-epoch run is a demonstration budget.",
        "",
        "![Learning curves](figures/02_learning_curves.png)",
        "",
        "## Test results",
        "",
        "| Model | RMSE | MAE | Direction accuracy | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + row["Model"]
            + " | "
            + " | ".join(number(row[key]) for key in ("RMSE", "MAE", "Direction accuracy", "Brier"))
            + " |"
        )
    lines += [
        "",
        "- **RMSE:** square root of mean squared error; lower is better and large errors count more.",
        "- **MAE:** average absolute error; lower is better. Both error metrics use decimal log-return units.",
        "- **Direction accuracy:** fraction of correct positive/nonpositive labels; 0.60 means 60%.",
        "- **Brier:** mean squared probability error; lower is better. Available only for probability outputs.",
        "- **—:** metric is not applicable; it is not a zero score.",
        "",
        "Zero return predicts no change. Historical mean and class-frequency baselines use training labels only. "
        "Compare direction accuracy with the majority-direction baseline, not automatically with 50%.",
        "",
        "![Model comparison](figures/03_model_comparison.png)",
        "",
        "## Actual versus predicted",
        "",
        "Use this plot to inspect whether models capture changes or mostly predict a constant value. "
        "Visual similarity alone does not establish useful accuracy.",
        "",
        "![Test predictions](figures/04_test_predictions.png)",
        "",
        "## Interpretation and next experiment",
        "",
        "Do not choose LSTM simply because it is the preferred architecture. Compare against the baselines, "
        "then repeat the same protocol for other symbols and seeds. Select settings using validation, "
        "not repeated inspection of test scores. Previously inspected test periods are not new unseen evidence.",
        "",
        "This compact report covers predictive accuracy. The optional trading simulation remains in the notebook. "
        "Unverified price adjustments and trading calendars limit conclusions; prediction accuracy is not proof of profit.",
        "",
        "## Files",
        "",
        "This Markdown file is the readable result. The four PNG files in `figures/` are the illustrations "
        "embedded above. Arrays, metric tables and fitted models remain in notebook memory. "
        "This export writes no CSV, JSON, model checkpoint or dataset snapshot. Rerunning it updates the same files.",
        "",
    ]
    report = output / "benchmark_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
