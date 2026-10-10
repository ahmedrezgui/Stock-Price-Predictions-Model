from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.benchmark import run_benchmark
from stock_forecasting.config import Config


def test_end_to_end_benchmark_alignment_reports_and_walkforward(bars, tmp_path, monkeypatch):
    # Real fitting/reporting, isolated files; no network or billable resources.
    monkeypatch.chdir(tmp_path)
    bars.iloc[:180].to_csv("TEST.csv", index=False)
    c = Config(symbol="TEST", sequence_length=5, hidden_size=4, epochs=1, patience=1, device="cpu")
    result = run_benchmark("TEST.csv", c, "report", seeds=(7,), tasks=("return",), losses=("huber",), folds=1)
    predictions = pd.read_csv("report/predictions.csv")
    groups = list(predictions.groupby("model"))
    for _, group in groups:
        np.testing.assert_array_equal(group.timestamp, groups[0][1].timestamp)
        assert (pd.to_datetime(group.target_timestamp) > pd.to_datetime(group.timestamp)).all()
    assert len(result["walk_forward"]) == 2
    for fold in result["walk_forward"]:
        assert fold["split"]["test_last_label"] < result["split"]["test_start"]
    assert len(result["models"]) == 7
    assert Path("report/evaluation_report.md").is_file()
    assert Path("report/figures/benchmark_comparison.png").stat().st_size > 1000
    json.loads(Path("report/benchmark_summary.json").read_text(), parse_constant=lambda value: pytest.fail(value))
    from stock_forecasting.summary import consolidate

    consolidate(["report"], "aggregate")
    assert Path("aggregate/paired_statistics.json").is_file()
    assert "TEST" in Path("aggregate/evaluation_report.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_benchmark("TEST.csv", c, "report", seeds=(7,), tasks=("return",), folds=0)


def test_cli_validate_features_snapshot_analyze_train_and_report(bars, tmp_path, monkeypatch, capsys):
    from stock_forecasting.cli import main

    monkeypatch.chdir(tmp_path)
    bars.to_csv("TEST.csv", index=False)
    c = Config(symbol="TEST", sequence_length=5, hidden_size=4, epochs=1, device="cpu")
    Path("config.json").write_text(json.dumps(c.to_dict()))
    for command in ("validate", "features", "snapshot", "analyze", "train"):
        monkeypatch.setattr(
            "sys.argv",
            ["stock-forecast", command, "--data", "TEST.csv", "--config", "config.json", "--output", command],
        )
        main()
    assert list(Path("artifacts/models").glob("*/weights.pt"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "stock-forecast",
            "train",
            "--family",
            "arima",
            "--data",
            "TEST.csv",
            "--config",
            "config.json",
            "--output",
            "arima",
        ],
    )
    main()
    assert Path("arima/arima.pkl").is_file()
    Path("input.txt").write_text("immutable")
    for action, file in (("upload", "input.txt"), ("download", "restored.txt")):
        monkeypatch.setattr("sys.argv", ["stock-forecast", "storage", action, "--file", file, "--key", "test.txt"])
        main()
    assert Path("restored.txt").read_text() == "immutable"


def test_azure_job_constructed_without_submission_to_service(tmp_path, monkeypatch):
    from stock_forecasting import azure

    client = Mock()
    monkeypatch.setattr(azure, "ml_client", lambda: client)
    azure.submit_training("configs/forecast.json", "azureml:biat-raw:1", "forecast-cpu")
    job = client.jobs.create_or_update.call_args.args[0]
    assert job.compute == "forecast-cpu"
    assert job.inputs["data"].path == "azureml:biat-raw:1"
    assert job.limits.timeout == 3600
    assert "${{outputs.models}}" in job.command
    monkeypatch.setattr("sys.argv", ["azure", "deploy"])
    with pytest.raises(SystemExit) as error:
        azure.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"horizon": 0},
        {"epochs": 0},
        {"train_fraction": 0.9, "validation_fraction": 0.2},
        {"task": "bad"},
        {"loss": "bad"},
        {"adjustment": "bad"},
        {"threshold": -1},
        {"feature_groups": ["future"]},
        {"dropout": 1},
        {"learning_rate": 0},
    ],
)
def test_configuration_validation(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes)
