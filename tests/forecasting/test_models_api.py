from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from stock_forecasting.api import cached_bundle, create_app
from stock_forecasting.artifacts import load_bundle, save_bundle
from stock_forecasting.benchmark import split_metadata
from stock_forecasting.data import fit_scaler, split_masks, transform
from stock_forecasting.models import ARIMAForecaster, NeuralForecaster, RecurrentNetwork


@pytest.mark.parametrize("family", ["lstm", "gru"])
@pytest.mark.parametrize("task", ["return", "direction", "multitask"])
def test_network_heads_and_roundtrip(prepared, tmp_path, family, task):
    c, frame, columns, samples, manifest = prepared
    c = replace(c, family=family, task=task)
    masks = split_masks(samples, 140, 190)
    scaler, rows = fit_scaler(frame, columns, samples, masks[0], c.sequence_length)
    x = transform(samples.x, scaler)
    network = RecurrentNetwork(len(columns), c)
    r, logits = network(torch.as_tensor(x[:3]))
    assert r.shape == logits.shape == (3,)
    model = NeuralForecaster(len(columns), c).fit(x[masks[0]], samples.y[masks[0]], x[masks[1]], samples.y[masks[1]])
    prediction = model.predict(x[masks[2]])
    assert (prediction["return"] is None) == (task == "direction")
    if prediction["probability"] is not None:
        assert ((prediction["probability"] >= 0) & (prediction["probability"] <= 1)).all()
    bundle = save_bundle(
        model,
        scaler,
        columns,
        {**manifest, "dataset_version": "test"},
        split_metadata(frame, samples, masks, rows),
        tmp_path,
    )
    restored, _, _ = load_bundle(bundle)
    for name, values in prediction.items():
        if values is not None:
            np.testing.assert_allclose(restored.predict(x[masks[2]])[name], values, atol=1e-7)
    (bundle / "weights.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(bundle)


def test_arima_does_not_observe_future(tmp_path):
    rng = np.random.default_rng(8)
    returns = rng.normal(0, 0.02, 90)
    model = ARIMAForecaster().fit(returns[:50], candidates=((1, 0, 0),))
    a = model.rolling_predict(returns, np.array([55, 60]), 49, 5)
    changed = returns.copy()
    changed[61:] = 99
    b = model.rolling_predict(changed, np.array([55, 60]), 49, 5)
    np.testing.assert_array_equal(a, b)
    assert all(entry["converged"] for entry in model.selection)
    path = model.save(tmp_path / "arima")
    restored = ARIMAForecaster.load(path)
    np.testing.assert_allclose(restored.rolling_predict(returns, np.array([55, 60]), 49, 5), a)


def test_api_predictions_validation_auth_and_versions(prepared, bars, tmp_path, monkeypatch):
    c, frame, columns, samples, manifest = prepared
    masks = split_masks(samples, 140, 190)
    scaler, rows = fit_scaler(frame, columns, samples, masks[0], c.sequence_length)
    model = NeuralForecaster(len(columns), c)
    model_dir, data_dir = tmp_path / "models", tmp_path / "data"
    data_dir.mkdir()
    bars.to_csv(data_dir / "TEST.csv", index=False)
    monkeypatch.setenv("FORECAST_MAX_AGE_DAYS", "10000")
    bundle = save_bundle(
        model,
        scaler,
        columns,
        {**manifest, "dataset_version": "test"},
        split_metadata(frame, samples, masks, rows),
        model_dir,
    )
    client = TestClient(create_app(model_dir, data_dir))
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert len(client.get("/models").json()) == 1
    request = {"symbol": "TEST", "horizon": 5}
    response = client.post("/v1/predict", json=request)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["probability_up"] is None and body["forecast_for"] is None
    assert body["forecast_horizon_sessions"] is None
    assert body["training_data_version"] == "test"
    from stock_forecasting.data import features, normalize

    expected_frame, _ = normalize(bars, c)
    expected_frame, expected_columns = features(expected_frame, c.feature_groups)
    expected_x = expected_frame[expected_columns].tail(c.sequence_length).to_numpy()[None]
    expected_prediction = model.predict(transform(expected_x, scaler))["return"][0]
    assert body["predicted_log_return"] == pytest.approx(float(expected_prediction), abs=1e-7)
    assert response.headers["x-request-id"]
    assert client.post("/predict", json=request).json() == body
    assert client.post("/predict", json={"symbol": "../secret"}).status_code == 422
    assert client.post("/predict", json={"symbol": "NO"}).status_code == 404
    assert client.post("/predict", json={**request, "model_version": "absent"}).status_code == 404
    monkeypatch.setenv("FORECAST_API_KEY", "local-test-key")
    assert client.post("/predict", json=request).status_code == 401
    assert client.post("/predict", json=request, headers={"X-API-Key": "local-test-key"}).status_code == 200
    monkeypatch.delenv("FORECAST_API_KEY")
    save_bundle(
        model,
        scaler,
        columns,
        {**manifest, "dataset_version": "test"},
        split_metadata(frame, samples, masks, rows),
        model_dir,
    )
    assert client.post("/predict", json=request).status_code == 409
    request["model_version"] = bundle.name
    assert client.post("/predict", json=request).status_code == 200
    monkeypatch.setenv("FORECAST_MAX_AGE_DAYS", "1")
    assert client.post("/predict", json=request).status_code == 422
    monkeypatch.setenv("FORECAST_MAX_AGE_DAYS", "10000")
    bars.iloc[:170].to_csv(data_dir / "TEST.csv", index=False)
    assert client.post("/predict", json=request).status_code == 422
    bars.to_csv(data_dir / "TEST.csv", index=False)
    Path(data_dir / "TEST.csv").unlink()
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", json=request).status_code == 503
    cached_bundle.cache_clear()


def test_empty_api_and_production_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("FORECAST_ENV", "production")
    monkeypatch.delenv("FORECAST_API_KEY", raising=False)
    client = TestClient(create_app(tmp_path, tmp_path))
    assert client.get("/ready").status_code == 503
    assert client.get("/models").status_code == 503


def test_importing_legacy_deployment_does_not_deploy(monkeypatch):
    import importlib.util
    from unittest.mock import Mock
    import azure.ai.ml

    client = Mock(side_effect=AssertionError("Cloud client must not be constructed at import"))
    monkeypatch.setattr(azure.ai.ml, "MLClient", client)
    spec = importlib.util.spec_from_file_location("legacy_deploy_no_side_effects", "scripts/deploy_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client.assert_not_called()
