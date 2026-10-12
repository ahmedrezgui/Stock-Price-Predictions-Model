"""Managed endpoint adapter consumes canonical completed OHLCV, not scaled tensors."""

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecasting.artifacts import load_bundle
from stock_forecasting.data import features, forecast_session, normalize, transform
from stock_forecasting.monitoring import database, database_path, save_forecast


def init():
    global bundle
    if not os.getenv("FORECAST_MONITOR_DB"):
        raise ValueError("Configure durable FORECAST_MONITOR_DB storage before deploying this adapter")
    with database(database_path()):
        pass
    paths = list(Path(os.environ["AZUREML_MODEL_DIR"]).rglob("checksums.json"))
    if len(paths) != 1:
        raise ValueError("Deploy exactly one verified model bundle")
    bundle = load_bundle(paths[0].parent)


def run(raw_data):
    payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    model, scaler, meta = bundle
    config = replace(model.config, start=None, end=None, as_of=None)
    frame, quality = normalize(pd.DataFrame(payload["observations"]), config)
    latest = frame.timestamp.iloc[-1]
    if (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - latest).days > int(
        os.getenv("FORECAST_MAX_AGE_DAYS", "14")
    ):
        raise ValueError("Latest completed observation is stale")
    origin_close = float(frame.close.iloc[-1])
    data_version = sha256(frame.to_csv(index=False).encode()).hexdigest()[:20]
    cutoff = max(pd.Timestamp(meta["split"][key]) for key in ("train_last_label", "validation_last_label"))
    if frame.timestamp.iloc[-1] < cutoff:
        raise ValueError("Historical origin predates labels used for training or model selection")
    frame, columns = features(frame, config.feature_groups)
    if columns != meta["feature_columns"]:
        raise ValueError("Incompatible feature ordering")
    x = frame[columns].tail(config.sequence_length).to_numpy()
    if len(x) != config.sequence_length or not np.isfinite(x).all():
        raise ValueError("Insufficient completed OHLCV history")
    prediction = model.predict(transform(x[None], scaler))
    result = {
        "symbol": config.symbol,
        "model": config.family,
        "model_version": meta["model_version"],
        "as_of": str(frame.timestamp.iloc[-1].date()),
        "horizon_observations": config.horizon,
        "forecast_horizon_observations": config.horizon,
        "horizon_basis": quality["horizon_basis"],
        "forecast_for": forecast_session(config, latest),
        "price_basis": config.adjustment,
        "data_version": data_version,
        "predicted_log_return": float(prediction["return"][0]) if prediction["return"] is not None else None,
        "probability_up": float(prediction["probability"][0]) if prediction["probability"] is not None else None,
        "training_data_version": meta["dataset"]["dataset_version"],
    }
    probability = result["probability_up"]
    result["direction"] = (
        "up"
        if (probability >= 0.5 if probability is not None else result["predicted_log_return"] > 0)
        else "nonpositive"
    )
    result["forecast_id"] = save_forecast(database_path(), result, config, origin_close)
    return result
