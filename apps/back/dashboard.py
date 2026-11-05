"""Dashboard data API; all forecasts execute on the remote forecasting service."""

import json

from fastapi import FastAPI, HTTPException

from market_data import CompanyNotFoundError, DataUnavailableError
from market_data.history import list_companies, load_closing_history
from stock_assistant.forecast_client import ForecastServiceError, get_prediction, get_models

app = FastAPI(title="StockWise market dashboard")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/companies")
def companies():
    return sorted(available_models())


def available_models():
    """Only advertise remote models matching the dashboard's prediction contract."""
    try:
        catalog = get_models()
    except ForecastServiceError:
        raise HTTPException(503, "Deployed model catalog is unavailable. Please retry shortly.") from None
    local = set(list_companies())
    return {entry["symbol"]: entry for entry in catalog if (
        entry.get("symbol") in local and entry.get("model") == "lstm"
        and entry.get("task") == "return" and entry.get("horizon") == 1
        and entry.get("model_version")
    )}


@app.get("/stock/{company}")
def stock(company: str):
    company = company.upper()
    model = available_models().get(company)
    if model is None:
        raise HTTPException(404, "No deployed LSTM forecast is available for this stock.")
    try:
        frame = load_closing_history(company, include_ohlcv=True)
    except CompanyNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None
    except DataUnavailableError:
        raise HTTPException(503, "Market history unavailable") from None
    result = {"columns": frame.columns.tolist(), "data": json.loads(frame.to_json(orient="records", date_format="iso"))}
    result["quality"] = frame.attrs.get("quality", {})
    result["model"] = model
    try:
        result["forecast"] = get_prediction(company, model_version=model["model_version"])
    except ForecastServiceError as exc:
        result["forecast_error"] = str(exc)
    return result
