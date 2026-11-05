import inspect
import pickle
from datetime import datetime

import joblib
import numpy as np
import pytest
import torch

from stock_forecasting.neural import GRUModel, IModel, LSTMModel


@pytest.mark.parametrize("model_class", [LSTMModel, GRUModel])
def test_learning_prediction_and_portable_serialization(model_class, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    features = rng.normal(size=(16, 5, 2)).astype(np.float64)
    targets = np.full(16, 0.5)
    date = datetime(2026, 10, 7)
    model = model_class("ATB")
    model.train(features, targets, date)
    predictions = model.predict(features)
    assert np.mean((predictions.reshape(-1) - targets) ** 2) < 0.01
    assert predictions.shape == ((16, 1) if model_class is LSTMModel else (16,))
    assert np.isfinite(predictions).all()
    assert model.predict(features[:1]).shape == ((1, 1) if model_class is LSTMModel else ())
    assert model.predict(features[:0]).shape == ((0, 1) if model_class is LSTMModel else (0,))
    np.testing.assert_array_equal(predictions, model.predict(features))
    assert model.get_stock_name() == "ATB"
    assert model.get_last_trained_date() == date
    restored = pickle.loads(pickle.dumps(model))
    assert next(restored.model.parameters()).device.type == "cpu"
    np.testing.assert_allclose(restored.predict(features), predictions, atol=1e-6)
    path = tmp_path / "model.pkl"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(features), predictions, atol=1e-6)


@pytest.mark.parametrize("model_class", [LSTMModel, GRUModel])
def test_invalid_data_and_untrained_prediction(model_class):
    model = model_class()
    with pytest.raises(RuntimeError, match="Train"):
        model.predict(np.zeros((1, 5, 2)))
    with pytest.raises(ValueError, match="shape"):
        model.train(np.zeros((3, 2)), np.zeros(3), datetime.now())
    with pytest.raises(ValueError, match="Targets"):
        model.train(np.zeros((3, 5, 2)), np.zeros((3, 2)), datetime.now())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("model_class", [LSTMModel, GRUModel])
def test_cuda_training(model_class):
    model = model_class()
    features = np.ones((4, 5, 2), dtype=np.float32)
    model.train(features, np.ones(4), datetime.now())
    assert next(model.model.parameters()).device.type == "cuda"
    assert np.isfinite(model.predict(features)).all()


@pytest.fixture(autouse=True)
def small_training_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("model_class", [LSTMModel, GRUModel])
def test_original_public_interface(model_class):
    assert model_class.__bases__ == (IModel,)
    assert list(inspect.signature(model_class.__init__).parameters) == ["self", "stock_name"]
    assert list(inspect.signature(model_class.train).parameters) == ["self", "features", "targets", "last_trained_date"]
    assert list(inspect.signature(model_class.predict).parameters) == ["self", "input_data"]
    model = model_class("ATB")
    assert set(vars(model)) == {"model", "stock_name", "last_trained_date"}
    assert isinstance(model.model, torch.nn.Module)
