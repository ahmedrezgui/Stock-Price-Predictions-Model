"""Forecasts come only from the prediction endpoint, preserving its evidence."""

import json
from langchain_core.tools import tool
from ..forecast_client import ForecastServiceError, get_prediction, get_models

FORECAST_DISCLAIMER = (
    "This statistical forecast uses historical market data, not news. "
    "It is uncertain and is not investment advice. Horizons follow the returned "
    "horizon_basis, not calendar days; implied_price is not a predicted daily price path."
)


@tool
def list_forecast_models() -> str:
    """List the forecasting endpoint's available model versions and evidence.

    Use this when prediction is unavailable or asks for an explicit model_version.
    Being listed does not itself mean the model is approved for deployment.
    """
    try:
        return json.dumps({"models": get_models()})
    except ForecastServiceError as exc:
        return json.dumps({"error": str(exc)})


@tool
def predict_stock(company: str, horizon: int = 1, model_version: str | None = None) -> str:
    """Call the LSTM return prediction API for a stock and observation horizon.

    The horizon must be 1, 5 or 20 observations (default 1).
    Preserve as_of, model_version, deployment_status, horizon_basis and all values
    when explaining the result. Relay the disclaimer. A horizon of 5 is a single
    cumulative forecast, not five daily predictions. Unavailable models cannot be
    replaced with invented forecasts. Use web_search for accompanying recent news.
    """
    # Gemini's protobuf tool schema cannot serialize numeric JSON-schema enums.
    # Expose an integer here; ForecastQuery enforces the allowed values before HTTP.
    try:
        forecast = get_prediction(company, horizon, model_version)
    except (ForecastServiceError, ValueError) as exc:
        return json.dumps({"error": str(exc), "note": "Use get_stock_data for historical data."})
    return json.dumps({**forecast, "disclaimer": FORECAST_DISCLAIMER}, allow_nan=False)
