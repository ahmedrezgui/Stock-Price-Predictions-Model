from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from stock_forecasting.config import Config
from stock_forecasting.notebook_benchmark import (
    benchmark_stock,
    comparison_tables,
    export_benchmark,
    model_choice_text,
    prepare_stock,
)


def test_multitask_benchmark_uses_same_origins_and_exports_only_report_figures(bars, tmp_path):
    csv = tmp_path / "TEST.csv"
    bars.to_csv(csv, index=False)
    config = Config(symbol="TEST", sequence_length=5, epochs=1, hidden_size=4, device="cpu")
    data = prepare_stock(csv, config, "2020-07-01", "2020-09-01")
    result = benchmark_stock(data, seeds=(7, 19))
    assert len(result["budgets"]) == 8
    assert set(result["budgets"].device) == {"cpu"}
    train, val, test = data["masks"]
    assert data["samples"].ends[train].max() < data["samples"].origins[val].min()
    assert data["samples"].ends[val].max() < data["samples"].origins[test].min()
    for entry in result["predictions"].values():
        for values in entry["prediction"].values():
            if values is not None:
                assert len(values) == int(test.sum())
    assert result["predictions"]["LSTM_return"]["prediction"]["probability"] is None
    assert result["predictions"]["LSTM_direction"]["prediction"]["return"] is None
    per_stock, overall = comparison_tables({"TEST": result})
    np.testing.assert_allclose(per_stock[per_stock.model == "historical_mean"].relative_rmse, 1)
    np.testing.assert_allclose(per_stock[per_stock.model == "historical_probability"].relative_brier, 1)
    assert set(overall.phase) == {"validation", "test"}
    report = export_benchmark({"TEST": result}, tmp_path / "report")
    assert report.is_file()
    files = [p for p in report.parent.rglob("*") if p.is_file()]
    assert len(files) == 8  # Report, overall figure, and six grouped stock figures.
    assert {p.suffix for p in files} == {".md", ".png"}
    assert "LSTM_direction" in report.read_text(encoding="utf-8")
    assert "Training before 2020-07-01" in report.read_text(encoding="utf-8")


def test_model_choice_uses_validation_even_when_test_prefers_another_model():
    overall = pd.DataFrame(
        [
            ("validation", "LSTM_return", 0.99),
            ("validation", "ARIMA", 1.01),
            ("test", "LSTM_return", 1.02),
            ("test", "ARIMA", 0.98),
        ],
        columns=["phase", "model", "relative_rmse"],
    )
    per_stock = overall.assign(symbol="TEST")
    choice = model_choice_text(per_stock, overall)
    assert "Provisional return-model choice: LSTM_return" in choice
    assert "lowest observed test error belongs to **ARIMA**" in choice
    assert "1.020000" in choice
    assert "not an unconditional production winner" in choice
    # A much better test score must still not select ARIMA.
    overall.loc[overall.phase == "test", "relative_rmse"] = [2.0, 0.1]
    assert "Provisional return-model choice: LSTM_return" in model_choice_text(per_stock, overall)


def test_cuda_request_never_silently_falls_back(bars, tmp_path, monkeypatch):
    csv = tmp_path / "TEST.csv"
    bars.to_csv(csv, index=False)
    data = prepare_stock(csv, Config(symbol="TEST", sequence_length=5), "2020-07-01", "2020-09-01")
    data["config"] = replace(data["config"], device="cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        benchmark_stock(data)
