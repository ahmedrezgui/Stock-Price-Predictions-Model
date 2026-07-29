"""Tool exposing the project's trained price-forecasting models to the assistant."""

import json
from typing import Optional

from langchain_core.tools import tool

from src.services import (
    CompanyNotFoundError,
    DataUnavailableError,
    ModelUnavailableError,
    get_prediction,
)

FORECAST_DISCLAIMER = (
    "This is the output of a statistical model trained on past closing prices only. "
    "It ignores news and market events, and is not investment advice."
)


@tool
def predict_stock(company: str, horizon_days: Optional[int] = None) -> str:
    """Forecast the closing price of a Tunisian stock with the project's trained model.

    Use this whenever the user asks where a stock is heading, or asks for a
    recommendation, so the advice rests on the model rather than on guesswork.
    Pair it with retrieve_news when recommending, since the model sees prices
    only. Always pass the disclaimer returned in "disclaimer" on to the user,
    and never present a forecast as a certainty.

    Args:
        company: the ticker symbol, e.g. "BIAT". Call list_available_companies
            first if you are unsure of the symbol.
        horizon_days: how many days ahead to forecast. Defaults to the model's
            trained window; accuracy degrades the further out you go.

    Returns:
        A JSON object with the last recorded close, the forecast values by date,
        and the disclaimer to relay.
    """
    try:
        forecast = get_prediction(company, horizon_days)
    except CompanyNotFoundError:
        return json.dumps({"error": f"No model or data for '{company}'."})
    except ModelUnavailableError:
        return json.dumps(
            {
                "error": f"The forecasting model for '{company}' is unavailable or out "
                "of date, so no forecast can be given right now. Historical data from "
                "get_stock_data is still available."
            }
        )
    except DataUnavailableError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Forecast failed for '{company}': {e}"})

    return json.dumps(
        {
            "company": forecast["company"],
            "last_recorded_close": forecast["last_close"],
            "last_recorded_date": forecast["last_date"].strftime("%Y-%m-%d"),
            "forecast": [
                {"date": date.strftime("%Y-%m-%d"), "predicted_close": round(value, 3)}
                for date, value in zip(forecast["dates"], forecast["values"])
            ],
            "disclaimer": FORECAST_DISCLAIMER,
        }
    )
