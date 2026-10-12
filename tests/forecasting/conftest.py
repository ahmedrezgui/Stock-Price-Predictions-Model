import numpy as np
import pandas as pd
import pytest

from stock_forecasting.config import Config
from stock_forecasting.data import features, normalize, sequences, targets


@pytest.fixture
def bars():
    rng = np.random.default_rng(123)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 240)))
    return pd.DataFrame(
        {
            "timestamp": pd.bdate_range("2020-01-01", periods=len(close)),
            "symbol": "TEST",
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(100, 10000, len(close)),
        }
    )


@pytest.fixture
def prepared(bars):
    config = Config(symbol="TEST", horizon=5, sequence_length=10, epochs=2, hidden_size=8, device="cpu")
    frame, manifest = normalize(bars, config)
    frame, columns = features(frame, config.feature_groups)
    frame = targets(frame, config.horizon)
    samples = sequences(frame, columns, config.sequence_length, config.horizon)
    return config, frame, columns, samples, manifest
