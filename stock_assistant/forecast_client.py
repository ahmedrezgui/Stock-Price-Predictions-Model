"""HTTP boundary to the independently deployed, governed forecasting service."""

import os
from typing import Literal
import httpx
from pydantic import BaseModel, ConfigDict, Field


class ForecastQuery(BaseModel):
    symbol: str = Field(pattern=r"^[A-Z0-9_.-]{1,20}$")
    model: Literal["lstm", "gru"] = "lstm"
    horizon: Literal[1, 5, 20] = 1
    task: Literal["return", "direction", "multitask"] = "return"
    model_version: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,120}$")


class ForecastResult(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    symbol: str
    model: str
    model_version: str
    as_of: str
    forecast_for: str | None
    forecast_horizon_observations: int
    horizon_basis: str
    predicted_log_return: float | None
    implied_price: float | None
    deployment_status: str


class ForecastServiceError(RuntimeError):
    pass


def forecast_url():
    url = os.getenv("ASSISTANT_FORECAST_URL", "").strip().rstrip("/")
    if not url:
        raise ForecastServiceError(
            "Remote forecasting endpoint is not configured. Set ASSISTANT_FORECAST_URL "
            "to the deployed forecasting API. No local model fallback is used."
        )
    return url


def get_prediction(company, horizon=1, model_version=None):
    query = ForecastQuery(
        symbol=company.upper(), horizon=1 if horizon is None else horizon, model_version=model_version
    )
    url = forecast_url()
    key = os.getenv("FORECAST_API_KEY")
    try:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            response = client.post(
                url + "/v1/predict", json=query.model_dump(exclude_none=True), headers={"X-API-Key": key} if key else {}
            )
            response.raise_for_status()
            result = ForecastResult.model_validate(response.json())
            if (
                result.symbol != query.symbol
                or result.forecast_horizon_observations != query.horizon
                or result.model != query.model
                or (query.model_version and result.model_version != query.model_version)
            ):
                raise ValueError("Forecast response does not match request")
            return result.model_dump()
    except httpx.HTTPStatusError as exc:
        raise ForecastServiceError(
            f"Forecast service returned HTTP {exc.response.status_code}. "
            "Check /models for available approved versions; historical data remains available."
        ) from None
    except httpx.RequestError:
        raise ForecastServiceError("Forecast service is unreachable or timed out.") from None
    except ValueError:
        raise ForecastServiceError("Forecast service returned an invalid or mismatched response.") from None


def get_models():
    url = forecast_url()
    key = os.getenv("FORECAST_API_KEY")
    try:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            response = client.get(url + "/models", headers={"X-API-Key": key} if key else {})
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, list) or any(not isinstance(entry, dict) for entry in result):
                raise ValueError("Invalid model catalog")
            return result
    except (httpx.HTTPError, ValueError):
        raise ForecastServiceError("Forecast model catalog is unavailable.") from None
