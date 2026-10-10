from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from stock_forecasting.config import Config
from stock_forecasting.data import (
    available_context,
    features,
    fit_scaler,
    normalize,
    sequences,
    split_masks,
    targets,
    transform,
)
from stock_forecasting.storage import AzureStore, LocalStore, retrieve, snapshot


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_labels_and_no_tail_direction(bars, horizon):
    result = targets(bars, horizon)
    np.testing.assert_allclose(
        result.target_return[:-horizon], np.log(bars.close.to_numpy()[horizon:] / bars.close.to_numpy()[:-horizon])
    )
    assert result.target_direction.tail(horizon).isna().all()
    assert result.target_timestamp.iloc[0] == bars.timestamp.iloc[horizon]


def test_future_mutation_does_not_change_past_features(bars):
    c = Config(symbol="TEST")
    a, _ = normalize(bars, c)
    original, columns = features(a, ["price", "volatility", "volume", "technical"])
    a.loc[150:, ["open", "high", "low", "close", "volume"]] *= 5
    changed, _ = features(a, ["price", "volatility", "volume", "technical"])
    pd.testing.assert_frame_equal(original.loc[:149, columns], changed.loc[:149, columns])
    assert original.return_lag_1.iloc[50] == original.log_return.iloc[49]
    assert not np.isinf(original[columns].to_numpy()).any()


def test_purged_boundaries_scaler_and_sequence_alignment(prepared):
    c, frame, columns, samples, _ = prepared
    train, val, test = split_masks(samples, 140, 190)
    assert samples.ends[train].max() < samples.origins[val].min()
    assert samples.ends[val].max() < samples.origins[test].min()
    scaler, rows = fit_scaler(frame, columns, samples, train, c.sequence_length)
    assert rows[-1] < samples.origins[val].min()
    np.testing.assert_allclose(scaler.mean_, frame.iloc[rows][columns].mean())
    changed = frame.copy()
    changed.loc[140:, columns] = 1e8
    other, _ = fit_scaler(changed, columns, samples, train, c.sequence_length)
    np.testing.assert_array_equal(scaler.mean_, other.mean_)
    for i in (0, len(samples.x) - 1):
        origin = samples.origins[i]
        np.testing.assert_allclose(samples.x[i, -1], frame.loc[origin, columns].to_numpy(dtype=float))
        assert samples.y[i] == frame.target_return.iloc[origin]
    assert np.isfinite(transform(samples.x, scaler)).all()
    with pytest.raises(ValueError, match="target"):
        sequences(frame, columns + ["target_return"], c.sequence_length, c.horizon)


def test_no_filling_duplicate_conflict_invalid_and_incomplete(bars):
    c = Config(symbol="TEST", as_of="2020-02-01")
    sparse = bars.drop(index=5)
    normalized, _ = normalize(sparse, c)
    assert bars.timestamp.iloc[5] not in normalized.timestamp.values
    assert normalized.timestamp.max() < pd.Timestamp("2020-02-01")
    duplicate = pd.concat([bars, bars.iloc[:1].assign(close=999)])
    with pytest.raises(ValueError, match="duplicate"):
        normalize(duplicate, c)
    for column, value in (("close", 0), ("volume", np.nan), ("high", 1)):
        bad = bars.copy()
        bad.loc[0, column] = value
        with pytest.raises(ValueError):
            normalize(bad, c)
    a, _ = normalize(bars.sample(frac=1, random_state=3), Config(symbol="TEST"))
    assert a.timestamp.is_monotonic_increasing


def test_adjustment_and_calendar(bars, tmp_path):
    c = Config(symbol="TEST", adjustment="adjusted")
    with pytest.raises(ValueError, match="adjusted_close"):
        normalize(bars, c)
    adjusted, _ = normalize(bars.assign(adjusted_close=bars.close / 2), c)
    np.testing.assert_allclose(adjusted.open, bars.open / 2)
    path = tmp_path / "sessions.csv"
    pd.DataFrame({"session": bars.timestamp}).to_csv(path, index=False)
    c = replace(c, adjustment="raw", session_file=str(path))
    _, meta = normalize(bars, c)
    assert meta["horizon_basis"] == "exchange_sessions"
    with pytest.raises(ValueError, match="Missing 1"):
        normalize(bars.drop(index=8), c)
    from stock_forecasting.data import forecast_session

    c = replace(c, horizon=5)
    future = pd.to_datetime(["2020-12-03", "2020-12-07", "2020-12-08", "2020-12-10", "2020-12-14"])
    pd.DataFrame({"session": list(bars.timestamp) + list(future)}).to_csv(path, index=False)
    assert forecast_session(c, bars.timestamp.iloc[-1]) == "2020-12-14"
    pd.DataFrame({"session": bars.timestamp}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="cover forecast"):
        forecast_session(c, bars.timestamp.iloc[-1])


def test_publication_delay(bars):
    context = pd.DataFrame({"available_at": [pd.Timestamp("2020-01-06")], "macro": [42]})
    joined = available_context(bars, context)
    assert joined.loc[joined.timestamp < "2020-01-06", "macro"].isna().all()
    assert joined.loc[joined.timestamp >= "2020-01-06", "macro"].eq(42).all()


def test_storage_immutable_lineage_and_traversal(bars, tmp_path):
    store = LocalStore(tmp_path)
    manifest = snapshot(bars, {"symbol": "TEST", "adjustment": "raw"}, store)
    assert snapshot(bars, {"symbol": "TEST", "adjustment": "raw"}, store) == manifest
    pd.testing.assert_frame_equal(retrieve(manifest, store), bars)
    with pytest.raises(FileExistsError):
        store.put(manifest["key"], b"different")
    for key in ("../secret", "/absolute", "C:/secret", "a\\b"):
        with pytest.raises(ValueError):
            store.get(key)
    broken = {**manifest, "sha256": "bad"}
    with pytest.raises(ValueError, match="checksum"):
        retrieve(broken, store)


def test_blob_immutable_and_cloud_failure():
    from unittest.mock import Mock
    from azure.core.exceptions import ResourceExistsError, ServiceRequestError

    client = Mock()
    container = client.get_container_client.return_value
    store = AzureStore("https://example.blob.core.windows.net", "datasets", client)
    store.put("test/data", b"a")
    container.upload_blob.assert_called_once_with(name="test/data", data=b"a", overwrite=False)
    container.upload_blob.side_effect = ResourceExistsError()
    container.download_blob.return_value.readall.return_value = b"a"
    store.put("test/data", b"a")
    with pytest.raises(FileExistsError):
        store.put("test/data", b"b")
    container.download_blob.side_effect = ServiceRequestError("network down")
    with pytest.raises(ServiceRequestError):
        store.get("test/data")
