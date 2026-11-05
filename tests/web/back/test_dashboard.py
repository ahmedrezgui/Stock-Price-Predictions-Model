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


def test_chart_endpoint_aggregates_months_without_requesting_prediction():
    history = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-02-02"]),
                            "cloture": [10., 12., 14.], "volume": [100, 200, 300]})
    with (patch("apps.back.dashboard.load_closing_history", return_value=history),
          patch("apps.back.dashboard.available_models", return_value={"AL": {}}),
          patch("apps.back.dashboard.get_prediction") as predict):
        response = TestClient(app).get("/stock/AL/chart?interval=Month&style=Line&sma=&momentum=None")
    assert response.status_code == 200
    assert response.json()["data"][0]["x"] == ["2026-01", "2026-02"]
    predict.assert_not_called()


def test_chart_rejects_invalid_controls_and_undeployed_stocks():
    client = TestClient(app)
    assert client.get("/stock/AL/chart?interval=Week").status_code == 422
    assert client.get("/stock/AL/chart?sma=-1").status_code == 422
    with patch("apps.back.dashboard.available_models", return_value={}):
        assert client.get("/stock/UNKNOWN/chart").status_code == 404


def test_market_list_uses_history_without_running_forecasts():
    history = pd.DataFrame({"date": pd.to_datetime(["2026-10-01", "2026-10-02"]),
                            "cloture": [100., 102.], "volume": [20., 30.]})
    with (patch("apps.back.dashboard.available_models", return_value={"BIAT": {}}),
          patch("apps.back.dashboard.load_closing_history", return_value=history),
          patch("apps.back.dashboard.get_prediction") as predict):
        rows = TestClient(app).get("/market").json()
    assert rows[0]["symbol"] == "BIAT"
    assert rows[0]["change_pct"] == pytest.approx(2)
    assert rows[0]["volume"] == 30
    predict.assert_not_called()


def test_news_filters_links_caches_search_and_hides_provider_errors():
    from apps.back.dashboard import _news_cache
    _news_cache.clear()
    with (patch("apps.back.dashboard.available_models", return_value={"BIAT": {}}),
          patch("stock_assistant.tools.web_search_tools.run_search", return_value={"documents": [
              {"title": "Update", "link": "https://example.com/news", "date": "2026-10-01"},
              {"title": "Unsafe", "link": "javascript:alert(1)"},
          ]}) as search):
        first = TestClient(app).get("/stock/BIAT/news").json()
        assert TestClient(app).get("/stock/BIAT/news").json() == first
        assert len(first["articles"]) == 1
        search.assert_called_once()
    _news_cache.clear()
    with (patch("apps.back.dashboard.available_models", return_value={"BIAT": {}}),
          patch("stock_assistant.tools.web_search_tools.run_search", return_value={"error": "secret-token", "documents": []})):
        response = TestClient(app).get("/stock/BIAT/news")
        assert response.json()["unavailable"] is True
        assert "secret-token" not in response.text
    _news_cache.clear()
