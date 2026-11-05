"""Dashboard data API; all forecasts execute on the remote forecasting service."""

import json
from datetime import datetime, timezone
from threading import Lock
from time import monotonic
from urllib.parse import urlparse
import pandas as pd
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from apps.back.trading_charts import build_chart
from market_data import CompanyNotFoundError, DataUnavailableError
from market_data.history import list_companies, load_closing_history
from stock_assistant.forecast_client import ForecastServiceError, get_prediction, get_models

app = FastAPI(title="StockWise market dashboard")
_news_cache = {}
_news_lock = Lock()


@app.get("/market")
def market():
    """One historical snapshot per deployed stock, without running inference."""
    rows = []
    for symbol in sorted(available_models()):
        try:
            frame = load_closing_history(symbol, include_ohlcv=True).sort_values("date")
            last = frame.iloc[-1]
            def finite(value):
                return float(value) if pd.notna(value) else None
            rows.append({"symbol": symbol, "close": float(last.cloture),
                         "change_pct": (float(last.cloture / frame.iloc[-2].cloture) - 1) * 100 if len(frame) > 1 else None,
                         "volume": finite(last.get("volume")), "as_of": last.date.strftime("%Y-%m-%d")})
        except (CompanyNotFoundError, DataUnavailableError, IndexError):
            rows.append({"symbol": symbol, "close": None, "change_pct": None, "volume": None, "as_of": None})
    return rows


@app.get("/stock/{company}/news")
def news(company: str):
    """Dated web-news links; cache searches for 15 minutes to bound provider usage."""
    company = company.upper()
    if company not in available_models():
        raise HTTPException(404, "No deployed model is available for this stock.")
    from stock_assistant.tools.web_search_tools import run_search
    with _news_lock:
        cached = _news_cache.get(company)
        if cached and monotonic() - cached[0] < 900:
            return cached[1]
        result = run_search(f'"{company}" entreprise cotée Bourse de Tunis actualités', max_results=8)
        articles, seen = [], set()
        for doc in result.get("documents", []):
            link = doc.get("link") or ""
            parsed = urlparse(link)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or link in seen:
                continue
            seen.add(link)
            published = pd.to_datetime(doc.get("date"), utc=True, errors="coerce")
            articles.append({"title": doc.get("title") or parsed.hostname, "link": link,
                             "date": published.isoformat() if pd.notna(published) else None,
                             "publisher": parsed.hostname.removeprefix("www.")})
        articles.sort(key=lambda a: a.get("date") or "", reverse=True)
        payload = {"articles": articles, "checked_at": datetime.now(timezone.utc).isoformat(),
                   "unavailable": bool(result.get("error"))}
        _news_cache[company] = (monotonic(), payload)
        return payload


@app.get("/stock/{company}/chart")
def chart(company: str, style: Literal["Candles", "Line"] = "Candles",
          interval: Literal["Day", "Month", "Year"] = "Day",
          period: Literal["1M", "3M", "6M", "1Y", "5Y", "All"] = "3M",
          momentum: Literal["RSI", "MACD", "None"] = "RSI",
          sma: str = "20,50", ema: str = "", bands: bool = False, volume: bool = True):
    """Render historical indicators without invoking the forecast endpoint."""
    def windows(value):
        try:
            values = list(dict.fromkeys(int(v) for v in value.split(",") if v))
        except ValueError:
            raise HTTPException(422, "Invalid indicator windows") from None
        if any(v not in {5, 10, 20, 50, 100, 200} for v in values):
            raise HTTPException(422, "Unsupported indicator window")
        return values
    sma_windows, ema_windows = windows(sma), windows(ema)
    company = company.upper()
    if company not in available_models():
        raise HTTPException(404, "No deployed LSTM forecast is available for this stock.")
    try:
        frame = load_closing_history(company, include_ohlcv=True)
    except CompanyNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None
    except DataUnavailableError:
        raise HTTPException(503, "Market history unavailable") from None
    figure = build_chart(frame, company, style, sma_windows, ema_windows, bands, momentum, period, volume, interval)
    return Response(figure.to_json(), media_type="application/json")


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
