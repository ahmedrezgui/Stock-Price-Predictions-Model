from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.config import Config
from stock_forecasting.monitoring import database, reconcile, run_monitoring, save_forecast, summarize


def issue(path, bars, index=10, horizon=1, version="v1", task="return", target=None, issued=None):
    c = Config(symbol="TEST", horizon=horizon, task=task)
    origin = pd.Timestamp(bars.timestamp.iloc[index])
    prediction = {
        "symbol": "TEST",
        "model": "lstm",
        "model_version": version,
        "as_of": str(origin.date()),
        "forecast_for": target,
        "forecast_horizon_observations": horizon,
        "horizon_basis": "exchange_sessions" if target else "observed_bars_unverified_sessions",
        "price_basis": "unverified",
        "predicted_log_return": 0.01 if task != "direction" else None,
        "probability_up": 0.7 if task != "return" else None,
        "direction": "up",
        "data_version": f"input-{index}",
        "training_data_version": "training",
    }
    forecast_id = save_forecast(
        path, prediction, c, float(bars.close.iloc[index]), issued or origin + pd.Timedelta(days=1)
    )
    return forecast_id, prediction, c


def row(path, forecast_id):
    with database(path) as connection:
        return dict(connection.execute("SELECT * FROM forecasts WHERE forecast_id=?", (forecast_id,)).fetchone())


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_maturity_uses_completed_observations_and_preserves_outcomes(tmp_path, bars, horizon):
    path = tmp_path / "forecasts.sqlite3"
    forecast_id, _, _ = issue(path, bars, horizon=horizon)
    bars.to_csv(tmp_path / "TEST.csv", index=False)
    target = bars.timestamp.iloc[10 + horizon]
    reconcile(path, tmp_path, now=target)  # Current day is not completed.
    assert row(path, forecast_id)["status"] == "pending"
    reconcile(path, tmp_path, now=target + pd.Timedelta(days=1))
    record = row(path, forecast_id)
    outcome = json.loads(record["outcome"])
    assert record["status"] == "matched"
    assert outcome["actual_log_return"] == pytest.approx(np.log(bars.close.iloc[10 + horizon] / bars.close.iloc[10]))
    assert outcome["target_date"] == str(target.date())
    reconcile(path, tmp_path, now=target + pd.Timedelta(days=2))
    assert row(path, forecast_id)["outcome"] == record["outcome"]


def test_concurrent_retries_do_not_weight_metrics_and_conflicts_fail(tmp_path, bars):
    path = tmp_path / "forecasts.sqlite3"
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: issue(path, bars)[0], range(12)))
    assert len(set(ids)) == 1
    assert summarize(path)["forecast_count"] == 1
    _, prediction, config = issue(path, bars)
    prediction["predicted_log_return"] = 0.03
    with pytest.raises(ValueError, match="already exists"):
        save_forecast(path, prediction, config, bars.close.iloc[10], now="2020-01-16")


@pytest.mark.parametrize(
    "change,reason",
    [("origin", "origin_price_revised"), ("outcome", "matched_outcome_revised"), ("missing", "origin_missing")],
)
def test_revisions_exclude_errors_without_erasing_original_evidence(tmp_path, bars, change, reason):
    path = tmp_path / "forecasts.sqlite3"
    forecast_id, _, _ = issue(path, bars)
    source = tmp_path / "TEST.csv"
    bars.to_csv(source, index=False)
    reconcile(path, tmp_path, now="2021-01-01")
    evidence = row(path, forecast_id)["outcome"]
    changed = bars.copy()
    if change == "missing":
        changed = changed.drop(index=10)
    else:
        changed.loc[10 if change == "origin" else 11, ["open", "high", "low", "close"]] *= 1.1
    changed.to_csv(source, index=False)
    reconcile(path, tmp_path, now="2021-01-02")
    record = row(path, forecast_id)
    assert record["status"] == "invalid" and record["reason"] == reason
    assert record["outcome"] == evidence
    assert summarize(path)["groups"][0]["latest"] is None
    bars.to_csv(source, index=False)
    reconcile(path, tmp_path, now="2021-01-03")
    assert row(path, forecast_id)["status"] == "matched"


def test_calendar_missing_target_and_late_issue_are_excluded(tmp_path, bars):
    path = tmp_path / "forecasts.sqlite3"
    target = str(bars.timestamp.iloc[15].date())
    forecast_id, _, _ = issue(path, bars, horizon=5, target=target)
    source = tmp_path / "TEST.csv"
    bars.drop(index=15).to_csv(source, index=False)
    reconcile(path, tmp_path, now="2021-01-01")
    assert row(path, forecast_id)["reason"] == "target_session_missing"
    bars.drop(index=12).to_csv(source, index=False)
    reconcile(path, tmp_path, now="2021-01-01")
    assert row(path, forecast_id)["reason"] == "session_horizon_mismatch"
    bars.to_csv(source, index=False)
    late, _, _ = issue(path, bars, version="late", issued="2021-01-01")
    reconcile(path, tmp_path, now="2021-01-02")
    assert row(path, late)["reason"] == "outcome_predates_issue"
    assert row(path, forecast_id)["status"] == "matched"


def test_missing_source_marks_previously_matched_results_unavailable(tmp_path, bars):
    path = tmp_path / "forecasts.sqlite3"
    forecast_id, _, _ = issue(path, bars)
    bars.to_csv(tmp_path / "TEST.csv", index=False)
    reconcile(path, tmp_path, now="2021-01-01")
    reconcile(path, tmp_path / "absent", now="2021-01-02")
    assert row(path, forecast_id)["status"] == "error"
    assert summarize(path)["issues"]
    assert summarize(path)["groups"][0]["latest"] is None


def test_rolling_metrics_group_versions_tasks_and_bound_window(tmp_path, bars):
    path = tmp_path / "forecasts.sqlite3"
    for index in range(10, 15):
        issue(path, bars, index=index, task="multitask")
    issue(path, bars, version="v2", task="direction")
    bars.to_csv(tmp_path / "TEST.csv", index=False)
    reconcile(path, tmp_path, now="2021-01-01")
    summary = summarize(path, window=3, min_observations=3)
    groups = {g["identity"]["model_version"]: g for g in summary["groups"]}
    first, second = groups["v1"], groups["v2"]
    assert len(first["history"]) == 5
    assert not first["history"][1]["sufficient_evidence"]
    assert first["latest"]["n"] == 3 and first["latest"]["sufficient_evidence"]
    actual = np.log(bars.close.iloc[13:16].to_numpy() / bars.close.iloc[12:15].to_numpy())
    assert first["latest"]["rmse"] == pytest.approx(np.sqrt(np.mean((0.01 - actual) ** 2)))
    assert first["latest"]["mae"] == pytest.approx(np.mean(abs(0.01 - actual)))
    assert first["latest"]["brier"] == pytest.approx(np.mean((0.7 - (actual > 0)) ** 2))
    assert first["latest"]["direction_accuracy"] == pytest.approx(np.mean(actual > 0))
    assert "rmse" not in second["latest"]
    assert not second["latest"]["sufficient_evidence"]
    with pytest.raises(ValueError):
        summarize(path, window=2, min_observations=3)


def test_adjusted_prices_and_zero_return_direction(tmp_path, bars):
    path = tmp_path / "forecasts.sqlite3"
    _, prediction, config = issue(tmp_path / "other.sqlite3", bars)
    config = replace(config, adjustment="adjusted")
    prediction["price_basis"] = "adjusted"
    bars["adjusted_close"] = bars.close / 2
    bars.loc[11, "adjusted_close"] = bars.adjusted_close.iloc[10]
    forecast_id = save_forecast(path, prediction, config, bars.adjusted_close.iloc[10], now="2020-01-16")
    bars.to_csv(tmp_path / "TEST.csv", index=False)
    reconcile(path, tmp_path, now="2021-01-01")
    outcome = json.loads(row(path, forecast_id)["outcome"])
    assert outcome["actual_log_return"] == 0 and not outcome["actual_up"]
    assert summarize(path)["groups"][0]["latest"]["direction_accuracy"] == 0


def test_monitoring_report_and_mlflow_artifacts(tmp_path, bars, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    path = tmp_path / "forecasts.sqlite3"
    issue(path, bars)
    bars.to_csv(tmp_path / "TEST.csv", index=False)
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    report = tmp_path / "report.md"
    summary = run_monitoring(path, tmp_path, report, uri=uri, now="2021-01-01")
    assert summary["forecast_count"] == 1
    assert "rmse" in report.read_text(encoding="utf-8")
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("stock-return-forecasting")
    runs = client.search_runs([experiment.experiment_id], "tags.workflow = 'prediction_monitoring'")
    assert len(runs) == 1 and runs[0].info.status == "FINISHED"
    assert runs[0].data.metrics["rolling.n"] == 1
    assert runs[0].data.params["model_version"] == "v1"
    assert client.list_artifacts(runs[0].info.run_id, "reports")


def test_unwritable_monitor_store_fails_prediction(prepared, bars, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from stock_forecasting.api import create_app
    from stock_forecasting.artifacts import save_bundle
    from stock_forecasting.benchmark import split_metadata
    from stock_forecasting.data import fit_scaler, split_masks
    from stock_forecasting.models import NeuralForecaster

    config, frame, columns, samples, manifest = prepared
    masks = split_masks(samples, 140, 190)
    scaler, indices = fit_scaler(frame, columns, samples, masks[0], config.sequence_length)
    save_bundle(
        NeuralForecaster(len(columns), config),
        scaler,
        columns,
        {**manifest, "dataset_version": "test"},
        split_metadata(frame, samples, masks, indices),
        tmp_path / "models",
    )
    bars.to_csv(tmp_path / "TEST.csv", index=False)
    monkeypatch.setenv("FORECAST_MAX_AGE_DAYS", "10000")
    client = TestClient(create_app(tmp_path / "models", tmp_path, monitor_path=tmp_path / "monitor.sqlite3"))
    response = client.post("/v1/predict", json={"symbol": "TEST", "horizon": 5})
    assert response.status_code == 200
    forecast_id = response.json()["forecast_id"]
    assert row(tmp_path / "monitor.sqlite3", forecast_id)["status"] == "pending"

    def failure(*args, **kwargs):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr("stock_forecasting.api.save_forecast", failure)
    assert client.post("/v1/predict", json={"symbol": "TEST", "horizon": 5}).status_code == 503


def test_azure_adapter_records_forecasts(prepared, bars, tmp_path, monkeypatch):
    import importlib.util
    from stock_forecasting.artifacts import save_bundle
    from stock_forecasting.benchmark import split_metadata
    from stock_forecasting.data import fit_scaler, split_masks
    from stock_forecasting.models import NeuralForecaster

    config, frame, columns, samples, manifest = prepared
    masks = split_masks(samples, 140, 190)
    scaler, indices = fit_scaler(frame, columns, samples, masks[0], config.sequence_length)
    save_bundle(
        NeuralForecaster(len(columns), config),
        scaler,
        columns,
        {**manifest, "dataset_version": "test"},
        split_metadata(frame, samples, masks, indices),
        tmp_path / "models",
    )
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("FORECAST_MAX_AGE_DAYS", "10000")
    monkeypatch.delenv("FORECAST_MONITOR_DB", raising=False)
    spec = importlib.util.spec_from_file_location("score_monitor_test", "infrastructure/azure/score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="FORECAST_MONITOR_DB"):
        module.init()
    path = tmp_path / "forecasts.sqlite3"
    monkeypatch.setenv("FORECAST_MONITOR_DB", str(path))
    module.init()
    result = module.run({"observations": bars.to_dict(orient="records")})
    assert row(path, result["forecast_id"])["status"] == "pending"
    assert result["horizon_observations"] == result["forecast_horizon_observations"] == 5
