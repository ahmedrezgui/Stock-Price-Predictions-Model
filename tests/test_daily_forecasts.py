from unittest.mock import Mock, patch

import pandas as pd
import pytest

from scripts.daily_forecasts import preflight, run, fresh_prediction, upload_chart_snapshot


def source(tmp_path):
    date = str((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).date())
    content = f"date,open,high,low,close,volume\n{date},10,12,9,11,100\n".encode()
    (tmp_path / "BIAT.csv").write_bytes(content)
    model = {"symbol": "BIAT", "model": "lstm", "task": "return", "horizon": 1, "model_version": "BIAT-v1"}
    return date, content, [model]


def test_preflight_requires_requested_freshness(tmp_path):
    date, _, catalog = source(tmp_path)
    assert preflight(tmp_path, catalog, date)[0]["as_of"] == date
    with pytest.raises(ValueError, match="before required"):
        preflight(tmp_path, catalog, str((pd.Timestamp(date) + pd.Timedelta(days=1)).date()))


def test_invalid_ohlc_never_passes_daily_publication(tmp_path):
    _, _, catalog = source(tmp_path)
    path = tmp_path / "BIAT.csv"
    path.write_text(path.read_text().replace(",10,12,9,11,100", ",15,12,9,11,100"))
    with pytest.raises(ValueError, match="OHLC range"):
        preflight(tmp_path, catalog)


def test_pipeline_freezes_data_and_calls_remote_prediction_after_uploads(tmp_path):
    date, content, catalog = source(tmp_path)
    rows = preflight(tmp_path, catalog)
    order = []
    def publish(path, symbol):
        assert path.read_bytes() == content
        order.append("publish")
    def upload(path, symbol):
        assert path.read_bytes() == content
        order.append("upload")
    def predict(row):
        order.append("predict")
        return {"as_of": date}
    result = run(tmp_path, rows, publish, upload, predict)
    assert order == ["publish", "upload", "predict"]
    assert result[0]["status"] == "success"


def test_source_change_prevents_any_publication(tmp_path):
    _, _, catalog = source(tmp_path)
    rows = preflight(tmp_path, catalog)
    (tmp_path / "BIAT.csv").write_text("changed")
    publish = Mock()
    with pytest.raises(ValueError, match="changed"):
        run(tmp_path, rows, publish, Mock())
    publish.assert_not_called()


def test_api_cache_must_refresh_before_forecast_is_accepted():
    predict = Mock(side_effect=[{"as_of": "2026-10-08"}, {"as_of": "2026-10-12"}])
    sleep = Mock()
    row = {"symbol": "BIAT", "model_version": "BIAT-v1", "as_of": "2026-10-12"}
    assert fresh_prediction(row, predict, sleep)["as_of"] == row["as_of"]
    sleep.assert_called_once_with(20)


def test_partial_failure_is_reported_without_requesting_forecast(tmp_path):
    _, _, catalog = source(tmp_path)
    rows = preflight(tmp_path, catalog)
    predict = Mock()
    result = run(tmp_path, rows, Mock(), Mock(side_effect=RuntimeError("upload")), predict)
    assert result[0]["status"] == "failed"
    assert result[0]["serving_published"]
    predict.assert_not_called()


def test_files_upload_does_not_inherit_forecast_blob_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://forecast.blob.core.windows.net")
    with patch("scripts.daily_forecasts.shutil.which", return_value="az"), patch("scripts.daily_forecasts.subprocess.run") as command:
        command.side_effect = [Mock(returncode=0, stdout='[{"value":"test-key"}]'), Mock(returncode=0)]
        upload_chart_snapshot("BIAT.csv", "BIAT", "dashboard", "stockwise-app", "rg")
    upload = command.call_args
    assert "AZURE_STORAGE_ACCOUNT_URL" not in upload.kwargs["env"]
    assert upload.kwargs["env"]["AZURE_STORAGE_ACCOUNT"] == "dashboard"
    assert "test-key" not in upload.args[0]
