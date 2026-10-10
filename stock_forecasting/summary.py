"""Aggregate completed fixed-horizon runs without selecting models by test scores."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .evaluation import block_interval, diebold_mariano
from .reporting import write_json


def consolidate(roots, output="reports"):
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    records, sources, paired, regime_rows = [], [], [], []
    for root in map(Path, roots):
        summary = json.loads((root / "benchmark_summary.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(root / "predictions.csv")
        by_model = {name: group.reset_index(drop=True) for name, group in predictions.groupby("model")}
        for name, group in by_model.items():
            if not name.startswith(("lstm_", "gru_")) or not group.predicted.notna().all():
                continue
            references = ["arima", "zero_return"]
            counterpart = name.replace("lstm_", "gru_", 1)
            if name.startswith("lstm_") and counterpart in by_model:
                references.append(counterpart)
            for reference in references:
                other = by_model[reference]
                if not group.timestamp.equals(other.timestamp) or not group.target_timestamp.equals(
                    other.target_timestamp
                ):
                    raise ValueError("Benchmark forecast-origin alignment mismatch")
                y, first, second = group.actual.to_numpy(), group.predicted.to_numpy(), other.predicted.to_numpy()
                h = summary["config"]["horizon"]
                paired.append(
                    {
                        "report": str(root),
                        "horizon": h,
                        "first": name,
                        "second": reference,
                        **diebold_mariano(y, first, second, h),
                        "mean_squared_loss_difference_95_block_ci": block_interval(
                            (y - first) ** 2 - (y - second) ** 2, h
                        ),
                    }
                )
        sources.append(
            {
                "report": str(root),
                "dataset": summary["dataset"],
                "config": summary["config"],
                "split": summary["split"],
                "selection": summary["selection"],
                "environment": summary["environment"],
            }
        )
        for name, model in summary["models"].items():
            for regime, metric in model.get("regimes", {}).items():
                if "regression" in metric:
                    regime_rows.append(
                        {
                            "horizon": summary["config"]["horizon"],
                            "family_task": name.split("_seed")[0],
                            "regime": regime,
                            "rmse": metric["regression"]["rmse"],
                        }
                    )
            metric = model["metrics"]
            records.append(
                {
                    "horizon_observations": summary["config"]["horizon"],
                    "model": name,
                    "family_task": name.split("_seed")[0],
                    "rmse": metric.get("regression", {}).get("rmse"),
                    "direction_accuracy": metric.get("direction", {}).get("accuracy"),
                    "brier": metric.get("probability", {}).get("brier"),
                    "net_strategy_return": model["backtest"]["net_cumulative_return"],
                    "buy_hold_gross_return": model["backtest"]["buy_hold_gross_return"],
                    "parameters": model.get("parameters"),
                    "parameter_bytes": model.get("parameter_bytes"),
                    "training_seconds": model.get("training_seconds"),
                    "batch_ms_per_sample": model.get("latency_ms_per_sample"),
                    "report": str(root),
                }
            )
    table = pd.DataFrame(records)
    write_json(output / "paired_statistics.json", paired)
    regimes = pd.DataFrame(regime_rows)
    if not regimes.empty:
        regimes.groupby(["horizon", "family_task", "regime"]).rmse.mean().to_csv(output / "regime_comparison.csv")
        first_h = min(regimes.horizon)
        values = regimes[
            (regimes.horizon == first_h)
            & regimes.family_task.isin(["arima", "zero_return", "lstm_return", "gru_return"])
        ]
        pivot = values.pivot_table(index="family_task", columns="regime", values="rmse", aggfunc="mean")
        ax = pivot.plot.barh(figsize=(10, 5), title=f"Historical regime subsets; horizon {first_h}")
        ax.set_xlabel("RMSE (decimal log return)")
        ax.figure.tight_layout()
        (output / "figures").mkdir(exist_ok=True)
        ax.figure.savefig(output / "figures" / "regime_comparison.png", dpi=140)
        plt.close(ax.figure)
    table.to_csv(output / "benchmark_summary.csv", index=False)
    groups = table.groupby(["horizon_observations", "family_task"])
    aggregates = groups[["rmse", "direction_accuracy", "brier", "net_strategy_return", "training_seconds"]].agg(
        ["mean", "std"]
    )
    aggregates.columns = ["_".join(c) for c in aggregates.columns]
    aggregates = aggregates.reset_index()
    aggregates.to_csv(output / "benchmark_aggregates.csv", index=False)
    write_json(
        output / "benchmark_summary.json",
        {
            "experiments": sources,
            "results": json.loads(table.to_json(orient="records")),
            "seed_aggregates": json.loads(aggregates.to_json(orient="records")),
        },
    )
    fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 5), squeeze=False)
    for ax, source in zip(axes[0], sources):
        h = source["config"]["horizon"]
        part = aggregates[
            (aggregates.horizon_observations == h)
            & aggregates.family_task.isin(["zero_return", "historical_mean", "arima", "lstm_return", "gru_return"])
        ]
        ax.barh(part.family_task, part.rmse_mean, xerr=part.rmse_std.fillna(0))
        ax.set_title(f"{h} observed bars - test RMSE")
        ax.set_xlabel("Decimal log return; error bar = seed std")
    fig.tight_layout()
    (output / "figures").mkdir(exist_ok=True)
    fig.savefig(output / "figures" / "horizon_comparison.png", dpi=140)
    plt.close(fig)
    source_descriptions = [
        f"{v['dataset']['symbol']} {v['dataset']['start']} to {v['dataset']['end']}; "
        f"h={v['config']['horizon']}, basis={v['dataset']['horizon_basis']}, "
        f"adjustment={v['dataset']['adjustment']}, epoch budget={v['config']['epochs']}"
        for v in sources
    ]
    lines = [
        "# Consolidated forecasting evaluation",
        "",
        *source_descriptions,
        "",
        "Seed-averaged scores below describe separate fitted models, not an ensemble. "
        "Validation selects loss settings; final test results never select seeds. "
        "Values are decimal log-return units. Consult source reports for seeds and exact split dates.",
        "",
        "| Horizon | Model/task | Mean RMSE | Seed std | Mean direction accuracy | Mean Brier |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    def fmt(v):
        return "n/a" if pd.isna(v) else f"{v:.6f}"

    for _, row in aggregates.iterrows():
        lines.append(
            f"| {row.horizon_observations} | {row.family_task} | "
            + " | ".join(fmt(row[k]) for k in ["rmse_mean", "rmse_std", "direction_accuracy_mean", "brier_mean"])
            + " |"
        )
    lines += [
        "",
        "![Horizon comparison](figures/horizon_comparison.png)",
        "",
        "A best seed cannot establish an architecture winner. Compare seed variation, training-derived baselines, "
        "walk-forward periods and paired HAC statistics. Paired p-values and dependence-aware loss-difference "
        "intervals are in paired_statistics.json; comparisons are exploratory and not multiple-testing corrected.",
        "",
        "![Regimes](figures/regime_comparison.png)",
        "",
        "Training-only statistical diagnostics are stored with each source report. Unit-root rejection is not "
        "proof of full stationarity or useful return predictability. Probability heads are uncalibrated.",
        "",
        "Economic simulations use next-open execution and nonoverlapping positions. Cost, threshold and holding "
        "assumptions are recorded in each configuration. Compare within horizon against the same-span gross "
        "buy/hold result. Unknown adjustments, sparse sessions and liquidity limit financial conclusions.",
        "",
        "See docs/model_comparison.md for the dated interpretation of the supplied BIAT experiments.",
        "",
        "## Detailed artifacts",
        "",
    ]
    for source in sources:
        root = Path(source["report"])
        rel = root.relative_to(output) if root.is_relative_to(output) else root
        lines.append(
            f"- [{root.name}]({rel.as_posix()}/evaluation_report.md): exact origins, forecasts, regimes, fold metrics, plots, training histories and versions."
        )
    lines += [
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python -m stock_forecasting.summary " + " ".join(str(v["report"]) for v in sources),
        "```",
        "",
        "This command aggregates existing runs. It does not refit or choose a model based on the holdout.",
    ]
    (output / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return aggregates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    print(consolidate(args.roots, args.output).to_string(index=False))
