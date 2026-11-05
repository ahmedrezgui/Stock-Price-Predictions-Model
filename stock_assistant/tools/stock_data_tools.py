"""Tools that read the project's own price history for Tunisian stocks."""

import json
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from market_data import (
    CompanyNotFoundError,
    DataUnavailableError,
)

from market_data.history import list_companies, load_company_df

# Enough rows for the model to describe a trend without flooding its context.
MAX_ROWS = 30

_COLUMN_LABELS = {
    "ouverture": "open",
    "haut": "high",
    "bas": "low",
    "cloture": "close",
    "volume": "volume",
}


@tool
def list_available_companies() -> str:
    """List the symbols of every Tunisian stock this assistant has data for.

    Use this when the user names a company you are unsure how to spell as a
    symbol, or when they ask what you can cover.

    Returns:
        A JSON object with a "companies" list of ticker symbols.
    """
    try:
        return json.dumps({"companies": sorted(list_companies())})
    except Exception as e:
        return json.dumps({"error": f"Could not list companies: {e}"})


@tool
def get_stock_data(
    company: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Get recorded daily prices for one Tunisian stock over a date range.

    This is the authoritative source for what a stock actually did: use it for
    any question about prices, movements, volume or trends, and never estimate
    those numbers yourself. It reports history only; for future values use
    predict_stock.

    Args:
        company: the ticker symbol, e.g. "BIAT". Call list_available_companies
            first if you are unsure of the symbol.
        start_date: optional start of the range, as YYYY-MM-DD. Defaults to the
            beginning of the record.
        end_date: optional end of the range, as YYYY-MM-DD. Defaults to the most
            recent day recorded.

    Returns:
        A JSON object with the period covered, summary statistics over that
        period, and up to 30 of the most recent daily rows within it.
    """
    try:
        df = load_company_df(company)
    except CompanyNotFoundError:
        try:
            available = ", ".join(sorted(list_companies()))
        except Exception:
            available = "unavailable"
        return json.dumps({"error": f"No data for '{company}'.", "available_companies": available})
    except DataUnavailableError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Could not load data for '{company}': {e}"})

    try:
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
    except Exception:
        return json.dumps({"error": "start_date and end_date must be formatted as YYYY-MM-DD."})

    if df.empty:
        return json.dumps({"error": f"No rows recorded for '{company}' in that date range."})

    df = df.sort_values("date")
    closes = df["cloture"]
    first_close = float(closes.iloc[0])
    last_close = float(closes.iloc[-1])

    rows = df.tail(MAX_ROWS)
    payload = {
        "company": company,
        "period": {
            "from": df["date"].iloc[0].strftime("%Y-%m-%d"),
            "to": df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "trading_days": int(len(df)),
        },
        "summary": {
            "first_close": first_close,
            "last_close": last_close,
            "change_pct": round((last_close - first_close) / first_close * 100, 2) if first_close else None,
            "min_close": float(closes.min()),
            "max_close": float(closes.max()),
            "mean_close": round(float(closes.mean()), 3),
            "total_volume": float(df["volume"].sum()),
        },
        "rows": [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                **{label: float(row[col]) for col, label in _COLUMN_LABELS.items() if col in df.columns},
            }
            for _, row in rows.iterrows()
        ],
    }
    if len(df) > MAX_ROWS:
        payload["rows_note"] = (
            f"Only the {MAX_ROWS} most recent of {len(df)} trading days are listed; "
            "the summary covers the whole period."
        )

    return json.dumps(payload)
