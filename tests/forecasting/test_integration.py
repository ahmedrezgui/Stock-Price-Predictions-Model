import numpy as np
import pytest
from stock_forecasting.config import Config
from stock_forecasting.models import create_forecaster
from stock_forecasting.training import train_candidate
from stock_forecasting.registry import read_registry, promote, rollback, identity
from stock_forecasting.artifacts import load_bundle
from stock_forecasting.neural import LSTMModel, GRUModel


@pytest.mark.parametrize("family,cls", [("lstm", LSTMModel), ("gru", GRUModel)])
def test_shared_public_classes_and_validation(family, cls):
    c = Config(family=family, epochs=1, hidden_size=4, device="cpu")
    model = create_forecaster(2, c)
    assert isinstance(model, cls)
    with pytest.raises(ValueError, match="explicit validation"):
        model.train(None, None, None)
    with pytest.raises(ValueError, match="nonempty"):
        model.fit(np.zeros((0, 3, 2)), np.zeros(0), np.zeros((1, 3, 2)), np.zeros(1))


def test_candidate_rejection_skip_refit_and_rollback(bars, tmp_path, monkeypatch):
    import stock_forecasting.training as training

    data = tmp_path / "TEST.csv"
    bars.to_csv(data, index=False)
    root, output = tmp_path / "models", tmp_path / "reports"
    c = Config(symbol="TEST", epochs=2, hidden_size=4, sequence_length=5, device="cpu")
    # Control gate scores to exercise both branches while doing real model training.
    scores = iter([2.0, 1.0])
    monkeypatch.setattr(training, "objective", lambda *args: next(scores))
    rejected = train_candidate(data, c, root, output)
    assert rejected["status"] == "rejected"
    assert read_registry(root)["active"] == {}
    assert train_candidate(data, c, root, output)["status"] == "unchanged"
    scores = iter([0.0, 1.0])
    accepted = train_candidate(data, c, root, output, force=True, refit_production=True)
    assert accepted["status"] == "candidate"
    assert read_registry(root)["active"] == {}
    # Explicit activation below tests the local API, not permission to release.
    promote(root, accepted["model_version"])
    model, _, meta = load_bundle(root / accepted["model_version"])
    assert isinstance(model, LSTMModel)
    assert meta["split"]["production_refit"]
    assert meta["split"]["research_holdout_retired"]
    from fastapi.testclient import TestClient
    from stock_forecasting.api import create_app

    monkeypatch.setenv("FORECAST_MAX_AGE_DAYS", "10000")
    monkeypatch.setenv("FORECAST_ENV", "local")
    monkeypatch.delenv("FORECAST_API_KEY", raising=False)
    with TestClient(create_app(root, tmp_path)) as client:
        response = client.post("/v1/predict", json={"symbol": "TEST"})
        assert response.status_code == 200, response.text
        assert response.json()["model_version"] == accepted["model_version"]
        assert response.json()["forecast_for"] is None
    from stock_forecasting.release import package_release

    with pytest.raises(ValueError, match="release gate"):
        package_release(tmp_path, tmp_path / "release")
    first = accepted["model_version"]
    scores = iter([0.0, 1.0])
    second = train_candidate(data, c, root, output, force=True, refit_production=True)
    promote(root, second["model_version"])
    assert read_registry(root)["active"][identity(c)] == second["model_version"]
    rollback(root, identity(c))
    assert read_registry(root)["active"][identity(c)] == first
    with pytest.raises(ValueError, match="identity"):
        promote(root, first, "wrong-family")
    (root / first / "weights.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        promote(root, first)
