from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from apps.back.dashboard import app
from stock_assistant.forecast_client import ForecastServiceError
from market_data.history import load_closing_history, load_company_df
from market_data import DataUnavailableError
import pytest


def test_dashboard_keeps_remote_forecast_separate_from_history():
    history = pd.DataFrame({"date": pd.to_datetime(["2026-10-08"]), "cloture": [228.0]})
    forecast = {"model_version": "AL-lstm-return-h1-test", "predicted_log_return": 0.001}
    with (
        patch("apps.back.dashboard.load_closing_history", return_value=history),
        patch("apps.back.dashboard.get_prediction", return_value=forecast) as predict,
    ):
        response = TestClient(app).get("/stock/AL")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1
    assert response.json()["forecast"] == forecast
    predict.assert_called_once_with("AL", model_version="AL-v1")


def test_dashboard_history_survives_prediction_outage():
    history = pd.DataFrame({"date": pd.to_datetime(["2026-10-08"]), "cloture": [228.0]})
    with (
        patch("apps.back.dashboard.load_closing_history", return_value=history),
        patch("apps.back.dashboard.get_prediction", side_effect=ForecastServiceError("Unavailable")),
    ):
        response = TestClient(app).get("/stock/AL")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1
    assert response.json()["forecast_error"] == "Unavailable"


def test_dashboard_import_does_not_require_legacy_cache():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.modules['valkey'] = None; import apps.back.dashboard"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_display_closes_survive_inconsistent_ohlc_without_relaxing_model_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_DATA_ROOT", str(tmp_path))
    (tmp_path / "AB.csv").write_text(
        "date,ouverture,haut,bas,cloture,volume\n2020-01-02,12,11,9,10,100\n"
    )
    with pytest.raises(DataUnavailableError, match="OHLC range inconsistency"):
        load_company_df("AB")
    with patch("apps.back.dashboard.get_prediction", side_effect=ForecastServiceError("Not deployed")):
        response = TestClient(app).get("/stock/AB")
    assert response.status_code == 200
    assert response.json()["data"][0]["cloture"] == 10
    assert response.json()["data"][0]["haut"] is None
    assert response.json()["quality"]["invalid_ohlc_rows"] == 1
    assert response.json()["forecast_error"] == "Not deployed"


@pytest.mark.parametrize("rows", [
    "2020-01-02,0\n", "2020-01-02,inf\n", "2020-01-02,invalid\n",
    "2020-01-02,10\n2020-01-02,11\n", "not-a-date,10\n",
])
def test_display_rejects_invalid_closes_and_dates(tmp_path, monkeypatch, rows):
    monkeypatch.setenv("ASSISTANT_DATA_ROOT", str(tmp_path))
    (tmp_path / "AB.csv").write_text("date,cloture\n" + rows)
    with pytest.raises(DataUnavailableError):
        load_closing_history("AB")


def test_display_parses_french_prices_and_excludes_unfinished_dates(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_DATA_ROOT", str(tmp_path))
    (tmp_path / "AB.csv").write_text("date;cloture\n02/01/2020;10,5\n02/01/2099;99\n")
    frame = load_closing_history("AB")
    assert frame.cloture.tolist() == [10.5]
    assert frame.date.dt.strftime("%Y-%m-%d").tolist() == ["2020-01-02"]


@pytest.fixture(autouse=True)
def deployed_catalog():
    catalog = [{"symbol": symbol, "model": "lstm", "task": "return", "horizon": 1, "model_version": symbol + "-v1"} for symbol in ["AL", "AB"]]
    with patch("apps.back.dashboard.get_models", return_value=catalog):
        yield


def test_companies_only_lists_compatible_deployed_models():
    catalog = [
        {"symbol": "AL", "model": "lstm", "task": "return", "horizon": 1, "model_version": "v1"},
        {"symbol": "AB", "model": "gru", "task": "return", "horizon": 1, "model_version": "v2"},
        {"symbol": "BIAT", "model": "lstm", "task": "return", "horizon": 5, "model_version": "v3"},
        {"symbol": "UNKNOWN", "model": "lstm", "task": "return", "horizon": 1, "model_version": "v4"},
    ]
    with patch("apps.back.dashboard.get_models", return_value=catalog):
        response = TestClient(app).get("/companies")
    assert response.json() == ["AL"]


def test_catalog_outage_does_not_advertise_untrained_stocks():
    with patch("apps.back.dashboard.get_models", side_effect=ForecastServiceError("offline")):
        assert TestClient(app).get("/companies").status_code == 503


def test_undeployed_stock_cannot_request_forecast():
    with patch("apps.back.dashboard.get_models", return_value=[]), patch("apps.back.dashboard.get_prediction") as predict:
        assert TestClient(app).get("/stock/AB").status_code == 404
        predict.assert_not_called()
