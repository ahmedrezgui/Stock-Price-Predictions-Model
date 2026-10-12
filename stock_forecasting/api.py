"""Standalone forecast API: no agent, data download, retraining or cloud mutation."""

from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import hmac
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from typing import Literal
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .artifacts import load_bundle
from .registry import read_registry
from .data import features, forecast_session, normalize, read_csv, transform
from .monitoring import database, database_path, save_forecast

logger = logging.getLogger("stock_forecasting.api")


class PredictionRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9_.-]+$")
    model: Literal["lstm", "gru"] = "lstm"
    horizon: Literal[1, 5, 20] = 1
    task: Literal["return", "direction", "multitask"] = "return"
    model_version: str | None = Field(default=None, max_length=120, pattern=r"^[A-Za-z0-9_-]+$")


class PredictionResponse(BaseModel):
    forecast_id: str | None = None
    symbol: str
    model: str
    model_version: str
    forecast_horizon_sessions: int | None
    forecast_horizon_observations: int
    horizon_basis: str
    as_of: str
    forecast_for: str | None
    predicted_log_return: float | None
    implied_price: float | None
    price_basis: str
    direction: str
    probability_up: float | None
    probability_calibration: str | None
    prediction_interval: list[float] | None = None
    data_version: str
    training_data_version: str
    status: str = "success"


@lru_cache(maxsize=32)
def cached_bundle(path: str):
    return load_bundle(path)


def predict_bundle(bundle_path, data_root, request, monitor_path=None):
    try:
        model, scaler, meta = cached_bundle(str(Path(bundle_path).resolve()))
        c = replace(model.config, start=None, end=None, as_of=None)
        frame, data_meta = normalize(read_csv(Path(data_root) / f"{request.symbol}.csv"), c)
        latest = frame.timestamp.iloc[-1]
        if (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - latest).days > int(
            os.getenv("FORECAST_MAX_AGE_DAYS", "14")
        ):
            raise HTTPException(422, "Latest completed observation is stale")
        information_cutoff = max(
            pd.Timestamp(meta["split"][key]) for key in ("train_last_label", "validation_last_label")
        )
        if latest < information_cutoff:
            raise ValueError("Historical origin predates labels used for training or model selection")
        engineered, columns = features(frame, c.feature_groups)
        if columns != meta["feature_columns"]:
            raise ValueError("Feature order/version mismatch")
        values = engineered[columns].tail(c.sequence_length).to_numpy()
        if len(values) != c.sequence_length or not np.isfinite(values).all():
            raise ValueError("Insufficient finite feature history")
        pred = model.predict(transform(values[None], scaler))
        predicted = float(pred["return"][0]) if pred["return"] is not None else None
        probability = float(pred["probability"][0]) if pred["probability"] is not None else None
        target_date = forecast_session(c, latest)
        implied = float(frame.close.iloc[-1] * np.exp(predicted)) if predicted is not None else None
        if implied is not None and not np.isfinite(implied):
            raise ValueError("Nonfinite implied price")
        response = PredictionResponse(
            symbol=request.symbol,
            model=c.family,
            model_version=meta["model_version"],
            forecast_horizon_sessions=c.horizon if target_date else None,
            forecast_horizon_observations=c.horizon,
            horizon_basis=data_meta["horizon_basis"],
            as_of=str(latest.date()),
            forecast_for=target_date,
            predicted_log_return=predicted,
            implied_price=implied,
            price_basis=c.adjustment,
            direction="up" if (probability >= 0.5 if probability is not None else predicted > 0) else "nonpositive",
            probability_up=probability,
            probability_calibration="uncalibrated" if probability is not None else None,
            data_version=sha256(frame.to_csv(index=False).encode()).hexdigest()[:20],
            training_data_version=meta["dataset"]["dataset_version"],
        )
        if monitor_path is not None:
            try:
                response.forecast_id = save_forecast(
                    monitor_path, response.model_dump(), c, float(frame.close.iloc[-1])
                )
            except (OSError, sqlite3.Error, ValueError):
                logger.exception("Forecast monitoring persistence failed")
                raise HTTPException(503, "Forecast could not be recorded; inspect monitoring storage") from None
        return response
    except FileNotFoundError:
        raise HTTPException(503, "Model artifact or historical data missing") from None
    except (ValueError, KeyError, RuntimeError) as exc:
        logger.warning("Forecast rejected: %s", type(exc).__name__)
        raise HTTPException(422, "Data, calendar or artifact incompatible; inspect service logs") from None


def create_app(model_root=None, data_root=None, monitor_path=None):
    root = Path(model_root or os.getenv("FORECAST_MODEL_ROOT", "artifacts/models"))
    data = Path(data_root or os.getenv("FORECAST_DATA_ROOT", "data/raw"))
    monitor = (
        monitor_path if monitor_path is not None else database_path(root.parent / "monitoring" / "forecasts.sqlite3")
    )
    app = FastAPI(title="Stock return forecasting", version="1.0.0")

    def monitoring_ready():
        if (
            os.getenv("FORECAST_ENV", "local") != "local"
            and monitor_path is None
            and not os.getenv("FORECAST_MONITOR_DB")
        ):
            raise HTTPException(503, "Configure durable FORECAST_MONITOR_DB storage before production serving")
        try:
            with database(monitor):
                pass
        except (OSError, sqlite3.Error):
            raise HTTPException(503, "Forecast monitoring storage unavailable") from None

    def authorize(x_api_key: str | None = Header(default=None)):
        key = os.getenv("FORECAST_API_KEY")
        if os.getenv("FORECAST_ENV", "local") != "local" and not key:
            raise HTTPException(503, "Service authentication is not configured")
        if key and not hmac.compare_digest(key, x_api_key or ""):
            raise HTTPException(401, "Invalid API key")

    def catalog():
        entries = []
        for path in sorted(root.glob("*/metadata.json")):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                c = meta["model"]["config"]
                entries.append(
                    {
                        "symbol": c["symbol"],
                        "model": c["family"],
                        "task": c["task"],
                        "horizon": c["horizon"],
                        "model_version": meta["model_version"],
                        "horizon_basis": meta["dataset"]["horizon_basis"],
                        "path": str(path.parent.resolve()),
                    }
                )
            except (ValueError, KeyError, OSError):
                logger.error("Invalid artifact metadata", extra={"artifact": path.parent.name})
        return entries

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.status_code, "message": exc.detail}, "request_id": request.state.request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "error": {"code": 422, "message": "Invalid request schema"},
                "request_id": request.state.request_id,
            },
        )

    @app.middleware("http")
    async def trace(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Forecast request failed", extra={"request_id": request.state.request_id})
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {"code": 500, "message": "Internal forecast error"},
                    "request_id": request.state.request_id,
                },
            )
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Latency-Ms"] = f"{elapsed:.3f}"
        logger.info(
            json.dumps(
                {
                    "request_id": request.state.request_id,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": elapsed,
                }
            )
        )
        return response

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        entries = catalog()
        if not entries:
            raise HTTPException(503, "No model bundles available")
        if os.getenv("FORECAST_ENV", "local") != "local" and not os.getenv("FORECAST_API_KEY"):
            raise HTTPException(503, "Service authentication is not configured")
        monitoring_ready()
        for entry in entries:
            try:
                cached_bundle(entry["path"])
                if not (data / f"{entry['symbol']}.csv").is_file():
                    raise ValueError("Missing data")
            except (ValueError, KeyError, RuntimeError, OSError):
                raise HTTPException(503, "A model or its data is unavailable") from None
        return {"status": "ready", "models": len(entries)}

    @app.get("/models", dependencies=[Depends(authorize)])
    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models():
        return [{k: v for k, v in e.items() if k != "path"} for e in catalog()]

    @app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(authorize)])
    @app.post("/v1/predict", response_model=PredictionResponse, dependencies=[Depends(authorize)])
    def predict(request: PredictionRequest):
        all_models = catalog()
        if request.symbol not in {e["symbol"] for e in all_models}:
            raise HTTPException(404, "Unknown symbol")
        matches = [
            e
            for e in all_models
            if all(e[k] == getattr(request, k) for k in ("symbol", "model", "horizon", "task"))
            and (request.model_version is None or e["model_version"] == request.model_version)
        ]
        if not matches:
            raise HTTPException(404, "Requested model version, horizon or task unavailable")
        if request.model_version is None:
            key = f"{request.symbol}-{request.model}-{request.task}-h{request.horizon}"
            active = read_registry(root)["active"].get(key)
            if (root / "active.json").exists() and not active:
                raise HTTPException(404, "No promoted model for this identity")
            if active:
                matches = [entry for entry in matches if entry["model_version"] == active]
                if not matches:
                    raise HTTPException(503, "Promoted model is unavailable")
        if len(matches) != 1:
            raise HTTPException(409, "Specify model_version from /models; multiple evaluated versions exist")
        monitoring_ready()
        return predict_bundle(matches[0]["path"], data, request, monitor_path=monitor)

    return app


app = create_app()
