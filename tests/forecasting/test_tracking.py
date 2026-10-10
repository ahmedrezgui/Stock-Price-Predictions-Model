import json
from pathlib import Path

import pytest

from stock_forecasting.config import Config
from stock_forecasting.benchmark import prepare, run_benchmark
from stock_forecasting.notebook_benchmark import prepare_stock, benchmark_stock, export_benchmark
from stock_forecasting.tracking import tracking_run, finite_metrics, TrackingRun
from stock_forecasting.training import train_candidate

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-forecast")
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment("test-forecast", artifact_location=(tmp_path / "artifacts").as_uri())
    return client, experiment


def test_disabled_tracking_and_failed_run(tmp_path, monkeypatch, tracking):
    client, experiment = tracking
    assert finite_metrics({"x": {"good": 2, "bad": float("nan"), "missing": None}, "array": [1]}) == {"x.good": 2.0}
    with pytest.raises(ValueError, match="fit failed"):
        with tracking_run(Config(), "test") as run:
            raise ValueError("fit failed")
    assert client.get_run(run.run_id).info.status == "FAILED"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    with tracking_run(Config(), "disabled") as disabled:
        assert disabled.run_id is None
        disabled.metrics({"a": 1})
    assert len(client.search_runs([experiment])) == 1


def test_notebook_and_training_share_dataset_metrics_and_artifacts(bars, tmp_path, tracking, monkeypatch):
    client, experiment = tracking
    csv = tmp_path / "TEST.csv"
    bars.to_csv(csv, index=False)
    config = Config(symbol="TEST", sequence_length=5, epochs=1, hidden_size=4, device="cpu")
    data = prepare_stock(csv, config, "2020-07-01", "2020-09-01")
    _, _, _, manifest = prepare(csv, config, tmp_path / "storage")
    assert data["manifest"]["dataset_version"] == manifest["dataset_version"]
    result = benchmark_stock(data, seeds=(7,))
    export_benchmark({"TEST": result}, tmp_path / "report")
    parent = result["tracking"].run_id
    children = client.search_runs([experiment], filter_string=f"tags.`mlflow.parentRunId` = '{parent}'")
    assert len(children) == 9
    children = [r for r in children if r.data.tags["workflow"] == "benchmark_model"]
    assert len(children) == 4
    for run in children:
        assert run.info.status == "FINISHED"
        assert run.data.params["dataset_version"] == manifest["dataset_version"]
        metric = "probability.brier" if run.data.params["task"] == "direction" else "regression.rmse"
        assert f"validation.{metric}" in run.data.metrics
        assert f"test.{metric}" in run.data.metrics
        assert client.get_metric_history(run.info.run_id, "training.train_loss")[0].step == 1
    assert client.get_run(parent).data.tags["report_exported"] == "true"
    assert client.list_artifacts(parent, "reports/figures/TEST")
    root, output = tmp_path / "models", tmp_path / "training"
    trained = train_candidate(csv, config, root, output)
    run = client.get_run(trained["mlflow_run_id"])
    assert run.data.params["dataset_version"] == manifest["dataset_version"]
    assert "validation.regression.rmse" in run.data.metrics
    assert not any(key.startswith("test.") for key in run.data.metrics)
    artifacts = client.list_artifacts(run.info.run_id, f"models/{trained['candidate_version']}")
    assert {Path(a.path).name for a in artifacts} == {"weights.pt", "metadata.json", "checksums.json"}
    cached = train_candidate(csv, config, root, output)
    assert cached["status"] == "unchanged"
    assert cached["model_version"] == trained["model_version"]
    assert cached["mlflow_run_id"] != trained["mlflow_run_id"]
    assert client.get_run(cached["mlflow_run_id"]).data.tags["status"] == "unchanged"

    def fail_upload(*args):
        raise RuntimeError("report upload failed")

    monkeypatch.setattr(TrackingRun, "artifacts", fail_upload)
    with pytest.raises(RuntimeError, match="report upload failed"):
        export_benchmark({"TEST": result}, tmp_path / "report")
    assert client.get_run(parent).info.status == "FAILED"
    assert client.get_run(parent).data.tags["report_exported"] == "failed"


def test_logging_failure_does_not_promote(bars, tmp_path, tracking, monkeypatch):
    from stock_forecasting import training

    client, experiment = tracking
    csv = tmp_path / "TEST.csv"
    bars.to_csv(csv, index=False)
    config = Config(symbol="TEST", sequence_length=5, epochs=1, hidden_size=4, device="cpu")

    def fail(*args):
        raise RuntimeError("artifact upload failed")

    monkeypatch.setattr(TrackingRun, "artifacts", fail)
    monkeypatch.setattr(training, "objective", lambda y, prediction: 0.0)
    with pytest.raises(RuntimeError, match="artifact upload failed"):
        train_candidate(csv, config, tmp_path / "models", tmp_path / "training")
    assert json.loads((tmp_path / "models/active.json").read_text())["active"] == {}
    assert client.search_runs([experiment])[0].info.status == "FAILED"


def test_research_cli_training_and_benchmark_are_tracked(bars, tmp_path, tracking, monkeypatch):
    import sys
    from stock_forecasting.cli import main

    client, experiment = tracking
    # The CLI's default storage path must remain confined to this test.
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "TEST.csv"
    bars.to_csv(csv, index=False)
    config = Config(symbol="TEST", sequence_length=5, epochs=1, hidden_size=4, device="cpu")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    for family in ("arima", "gru"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "forecast",
                "train",
                "--data",
                str(csv),
                "--config",
                str(config_path),
                "--family",
                family,
                "--output",
                str(tmp_path / family),
            ],
        )
        main()
    runs = client.search_runs([experiment])
    assert {r.data.params["family"] for r in runs} == {"arima", "gru"}
    assert all("validation.regression.rmse" in r.data.metrics for r in runs)
    summary = run_benchmark(
        csv,
        config,
        tmp_path / "research",
        seeds=(7,),
        tasks=("return",),
        losses=("huber",),
        folds=0,
        artifacts=tmp_path / "models",
    )
    run = client.get_run(summary["mlflow_run_id"])
    assert run.info.status == "FINISHED"
    assert client.list_artifacts(run.info.run_id, "reports")
    children = client.search_runs([experiment], filter_string=f"tags.`mlflow.parentRunId` = '{run.info.run_id}'")
    assert len(children) == 7  # Four naive baselines, ARIMA, LSTM, GRU.
    assert all(r.info.status == "FINISHED" for r in children)


def test_source_changes_change_dataset_version(bars, tmp_path):
    csv = tmp_path / "TEST.csv"
    config = Config(symbol="TEST", sequence_length=5)
    bars.to_csv(csv, index=False)
    before = prepare_stock(csv, config, "2020-07-01", "2020-09-01")["manifest"]
    bars.loc[0, "volume"] += 1
    bars.to_csv(csv, index=False)
    after = prepare_stock(csv, config, "2020-07-01", "2020-09-01")["manifest"]
    assert before["dataset_version"] != after["dataset_version"]
    assert before["source_file_sha256"] != after["source_file_sha256"]


def test_refit_keeps_candidate_evaluation_separate(bars, tmp_path, tracking, monkeypatch):
    from stock_forecasting import training

    client, _ = tracking
    csv = tmp_path / "TEST.csv"
    bars.to_csv(csv, index=False)
    config = Config(symbol="TEST", sequence_length=5, epochs=1, hidden_size=4, device="cpu")
    # Force the acceptance branch; this test checks lineage, not model performance.
    monkeypatch.setattr(training, "objective", lambda y, prediction: 0.0)
    result = train_candidate(csv, config, tmp_path / "models", tmp_path / "training", refit_production=True)
    assert result["candidate_version"] != result["model_version"]
    run = client.get_run(result["mlflow_run_id"])
    assert run.data.tags["holdout_used"] == "True"
    assert not any(k.startswith("test.") for k in run.data.metrics)
    versions = client.list_artifacts(run.info.run_id, "models")
    assert {Path(v.path).name for v in versions} == {result["candidate_version"], result["model_version"]}
