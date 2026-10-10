"""Managed endpoint adapter consumes canonical completed OHLCV, not scaled tensors."""

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecasting.artifacts import load_bundle
from stock_forecasting.data import features, normalize, transform


def init():
    global bundle
    paths = list(Path(os.environ["AZUREML_MODEL_DIR"]).rglob("checksums.json"))
    if len(paths) != 1:
        raise ValueError("Deploy exactly one verified model bundle")
    bundle = load_bundle(paths[0].parent)


def run(raw_data):
    payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    model, scaler, meta = bundle
    config = replace(model.config, start=None, end=None, as_of=None)
    frame, _ = normalize(pd.DataFrame(payload["observations"]), config)
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
    return {
        "model_version": meta["model_version"],
        "as_of": str(frame.timestamp.iloc[-1].date()),
        "horizon_observations": config.horizon,
        "horizon_basis": meta["dataset"]["horizon_basis"],
        "predicted_log_return": float(prediction["return"][0]) if prediction["return"] is not None else None,
        "probability_up": float(prediction["probability"][0]) if prediction["probability"] is not None else None,
        "training_data_version": meta["dataset"]["dataset_version"],
    }
