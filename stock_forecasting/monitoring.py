"""Persist issued forecasts and reconcile daily outcomes without refitting models."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .config import Config
from .data import normalize, read_csv


def utc_now(now=None):
    value = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if pd.isna(value):
        raise ValueError("Invalid monitoring time")
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def database_path(default="artifacts/monitoring/forecasts.sqlite3"):
    return Path(os.getenv("FORECAST_MONITOR_DB", default))


@contextmanager
def database(path):
    """Short transactions; sqlite serializes concurrent writers on a local disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS forecasts (
                    forecast_id TEXT PRIMARY KEY, identity TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    outcome TEXT, reason TEXT, checked_at TEXT
                )"""
            )
            yield connection
    finally:
        connection.close()


def save_forecast(path, prediction, config, origin_close, now=None):
    """Keep the first issued forecast per origin/version, so retries don't weight metrics."""
    issued = utc_now(now)
    origin = pd.Timestamp(prediction["as_of"])
    if origin >= issued.tz_localize(None).normalize():
        raise ValueError("Forecast origin must be a completed daily bar")
    values = [origin_close] + [
        prediction[k] for k in ("predicted_log_return", "probability_up") if prediction[k] is not None
    ]
    if not np.isfinite(values).all() or origin_close <= 0:
        raise ValueError("Invalid forecast values")
    probability = prediction["probability_up"]
    if probability is not None and not 0 <= probability <= 1:
        raise ValueError("Invalid forecast probability")
    if prediction["predicted_log_return"] is None and probability is None:
        raise ValueError("Forecast has no predictions")
    identity = json.dumps(
        [
            config.symbol,
            prediction["model_version"],
            config.family,
            config.task,
            config.horizon,
            prediction["horizon_basis"],
            config.adjustment,
            prediction["as_of"],
        ],
        separators=(",", ":"),
    )
    forecast_id = sha256(identity.encode()).hexdigest()
    payload = {
        **prediction,
        "forecast_id": forecast_id,
        "issued_at": issued.isoformat(),
        "origin_close": float(origin_close),
        "task": config.task,
        "source": config.source,
    }
    payload.pop("status", None)
    with database(path) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO forecasts (forecast_id, identity, payload) VALUES (?, ?, ?)",
            (forecast_id, identity, json.dumps(payload, allow_nan=False)),
        )
        saved = json.loads(
            connection.execute("SELECT payload FROM forecasts WHERE forecast_id = ?", (forecast_id,)).fetchone()[0]
        )
        # The same identity must never refer to a different input or prediction.
        for key in ("data_version", "origin_close", "predicted_log_return", "probability_up", "forecast_for"):
            if saved[key] != payload[key]:
                raise ValueError("Forecast for this origin/version already exists with different data or values")
    return forecast_id


def match_outcome(forecast, frame, now):
    """Return status/reason/outcome; never guess an exchange calendar."""
    dates = frame.timestamp.dt.strftime("%Y-%m-%d")
    origins = np.flatnonzero(dates == forecast["as_of"])
    if len(origins) != 1:
        return "invalid", "origin_missing", None
    origin = int(origins[0])
    if not np.isclose(frame.close.iloc[origin], forecast["origin_close"], rtol=1e-10, atol=1e-12):
        return "invalid", "origin_price_revised", None
    target = origin + forecast["forecast_horizon_observations"]
    expected = forecast["forecast_for"]
    if expected and dates.iloc[-1] >= expected and expected not in set(dates):
        return "invalid", "target_session_missing", None
    if target >= len(frame):
        return "pending", "horizon_not_observed", None
    target_date = dates.iloc[target]
    if expected and target_date != expected:
        return "invalid", "session_horizon_mismatch", None
    if target_date < str(pd.Timestamp(forecast["issued_at"]).date()):
        return "invalid", "outcome_predates_issue", None
    target_close = float(frame.close.iloc[target])
    actual = float(np.log(target_close / forecast["origin_close"]))
    return (
        "matched",
        None,
        {
            "target_date": target_date,
            "target_close": target_close,
            "actual_log_return": actual,
            "actual_up": actual > 0,
            "matched_at": now.isoformat(),
            "same_day_issue": target_date == str(pd.Timestamp(forecast["issued_at"]).date()),
        },
    )


def reconcile(path, data_root, now=None):
    """Recheck even matched rows: revised outcomes are excluded, never silently replaced."""
    now = utc_now(now)
    cache = {}
    with database(path) as connection:
        rows = connection.execute("SELECT * FROM forecasts").fetchall()
        for row in rows:
            forecast = json.loads(row["payload"])
            key = (forecast["symbol"], forecast["price_basis"], forecast["source"])
            if key not in cache:
                try:
                    source = Path(data_root) / f"{key[0]}.csv"
                    # Calendar target dates were frozen at issue time; local calendar files may move.
                    frame, _ = normalize(
                        read_csv(source),
                        Config(
                            symbol=key[0],
                            adjustment=key[1],
                            source=key[2],
                            as_of=str(now.tz_localize(None).date()),
                        ),
                    )
                    digest = sha256(frame.to_csv(index=False).encode()).hexdigest()[:20]
                    cache[key] = (frame, digest)
                except (OSError, ValueError, KeyError) as exc:
                    cache[key] = f"{type(exc).__name__}: {exc}"
            data = cache[key]
            previous = json.loads(row["outcome"]) if row["outcome"] else None
            if isinstance(data, str):
                status, reason, outcome = "error", data, None
            else:
                frame, digest = data
                status, reason, outcome = match_outcome(forecast, frame, now)
                if outcome:
                    outcome["outcome_data_version"] = digest
                if (
                    previous
                    and outcome
                    and any(previous[k] != outcome[k] for k in ("target_date", "target_close", "actual_log_return"))
                ):
                    status, reason = "invalid", "matched_outcome_revised"
                # Original outcome evidence remains immutable, including when source data disappears.
            connection.execute(
                "UPDATE forecasts SET status=?, reason=?, outcome=?, checked_at=? WHERE forecast_id=?",
                (
                    status,
                    reason,
                    json.dumps(previous or outcome, allow_nan=False) if previous or outcome else None,
                    now.isoformat(),
                    row["forecast_id"],
                ),
            )
    return len(rows)


GROUP_KEYS = (
    "symbol",
    "model_version",
    "model",
    "task",
    "forecast_horizon_observations",
    "horizon_basis",
    "price_basis",
)


def error_metrics(rows):
    actual = np.array([r["actual_log_return"] for r in rows])
    result = {"n": len(rows), "same_day_issue_n": sum(r["same_day_issue"] for r in rows)}
    returns = [
        (r["predicted_log_return"], r["actual_log_return"]) for r in rows if r["predicted_log_return"] is not None
    ]
    if returns:
        residual = np.array([p - y for p, y in returns])
        result.update(
            return_n=len(returns),
            rmse=float(np.sqrt(np.mean(residual**2))),
            mae=float(np.mean(np.abs(residual))),
            bias=float(np.mean(residual)),
            zero_return_rmse=float(np.sqrt(np.mean([y**2 for _, y in returns]))),
        )
    probabilities = [(r["probability_up"], r["actual_up"]) for r in rows if r["probability_up"] is not None]
    if probabilities:
        result.update(probability_n=len(probabilities), brier=float(np.mean([(p - y) ** 2 for p, y in probabilities])))
    result["direction_accuracy"] = float(np.mean([(r["direction"] == "up") == (y > 0) for r, y in zip(rows, actual)]))
    return result


def summarize(path, window=30, min_observations=20):
    if not 1 <= min_observations <= window:
        raise ValueError("Require 1 <= min_observations <= window")
    with database(path) as connection:
        records = connection.execute("SELECT * FROM forecasts ORDER BY forecast_id").fetchall()
    groups = {}
    issues = []
    for record in records:
        forecast = json.loads(record["payload"])
        key = tuple(forecast[k] for k in GROUP_KEYS)
        group = groups.setdefault(key, {"identity": dict(zip(GROUP_KEYS, key)), "counts": {}, "rows": []})
        status = record["status"]
        group["counts"][status] = group["counts"].get(status, 0) + 1
        if status == "matched":
            group["rows"].append({**forecast, **json.loads(record["outcome"])})
        elif status in ("invalid", "error"):
            issues.append(
                {"forecast_id": record["forecast_id"], "symbol": forecast["symbol"], "reason": record["reason"]}
            )
    result = {
        "window": window,
        "min_observations": min_observations,
        "forecast_count": len(records),
        "groups": [],
        "issues": issues,
    }
    for group in groups.values():
        rows = sorted(group.pop("rows"), key=lambda r: (r["target_date"], r["as_of"]))
        history = []
        for end in range(1, len(rows) + 1):
            selected = rows[max(0, end - window) : end]
            history.append(
                {
                    "through": selected[-1]["target_date"],
                    "from": selected[0]["target_date"],
                    "sufficient_evidence": len(selected) >= min_observations,
                    **error_metrics(selected),
                }
            )
        group.update(latest=history[-1] if history else None, history=history, matched_forecasts=rows)
        result["groups"].append(group)
    return result


def write_report(summary, output):
    lines = [
        "# Prediction monitoring",
        "",
        f"Updated: {summary['checked_at']}",
        "",
        f"Rolling window: last {summary['window']} matched forecast origins per model identity. "
        f"Minimum evidence: {summary['min_observations']}. Stored forecasts: {summary['forecast_count']}.",
        "",
        "Errors use log returns; zero-return RMSE is the no-change baseline. "
        "Direction accuracy and Brier score use up = actual return > 0. "
        "Insufficient samples are descriptive only. This report does not authorize deployment.",
        "",
    ]
    if not summary["groups"]:
        lines.append("No forecasts recorded yet. Serve predictions before evaluating monitoring performance.")
    for group in summary["groups"]:
        identity = group["identity"]
        lines.extend(
            [
                f"## {identity['symbol']} — {identity['model_version']}",
                "",
                "Identity: "
                + ", ".join(f"{k}={v}" for k, v in identity.items() if k not in ("symbol", "model_version")),
                "",
                "Counts: " + ", ".join(f"{k}={v}" for k, v in group["counts"].items()),
                "",
            ]
        )
        if group["latest"]:
            lines.extend(["| Latest rolling metric | Value |", "| --- | --- |"])
            for key, value in group["latest"].items():
                lines.append(f"| {key} | {value:.6g} |" if isinstance(value, float) else f"| {key} | {value} |")
            lines.append("")
        else:
            lines.extend(["No eligible matched outcomes yet.", ""])
    if summary["issues"]:
        lines.extend(["## Data issues excluded from metrics", ""])
        lines.extend(f"- {r['symbol']} / {r['forecast_id'][:12]}: {r['reason']}" for r in summary["issues"])
    lines.extend(
        [
            "",
            "Same-day issuance is counted separately: daily CSV dates cannot establish whether a request "
            "preceded the exchange close. These are daily-bar prediction errors, not verified live trading results. "
            "Unverified observed-bar horizons cannot detect missing exchange sessions. Overlapping longer horizons "
            "are correlated; sample counts are not independent evidence.",
            "",
        ]
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def run_monitoring(path, data_root, output, window=30, min_observations=20, uri=None, now=None):
    from .tracking import tracking_run

    if not 1 <= min_observations <= window:
        raise ValueError("Require 1 <= min_observations <= window")
    checked = utc_now(now)
    reconcile(path, data_root, checked)
    summary = summarize(path, window, min_observations)
    summary["checked_at"] = checked.isoformat()
    write_report(summary, output)
    for group in summary["groups"]:
        identity = group["identity"]
        config = Config(
            symbol=identity["symbol"],
            family=identity["model"],
            task=identity["task"],
            horizon=identity["forecast_horizon_observations"],
            adjustment=identity["price_basis"],
        )
        with tracking_run(config, "prediction_monitoring", uri=uri) as run:
            run.params({**identity, "window": window, "min_observations": min_observations})
            run.tags({"model_version": identity["model_version"], "monitoring_checked_at": checked.isoformat()})
            run.metrics(group["counts"], "counts")
            for step, point in enumerate(group["history"]):
                run.metrics(point, "rolling", step)
            run.document(group, "monitoring.json")
            run.artifacts(output, "reports")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=database_path())
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--report", type=Path, default=Path("reports/monitoring/monitoring_report.md"))
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--min-observations", type=int, default=20)
    parser.add_argument("--mlflow-uri")
    args = parser.parse_args()
    result = run_monitoring(args.database, args.data, args.report, args.window, args.min_observations, args.mlflow_uri)
    print(f"Forecasts: {result['forecast_count']}; data issues: {len(result['issues'])}; report: {args.report}")
    if result["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
