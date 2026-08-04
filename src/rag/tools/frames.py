"""
Shared loading, date-window and error helpers for the stock-data tools.

Every tool that reads price history needs the same three things: a frame narrowed
to a date range, a way to say "that date made no sense" separately from "there is
no data there", and a consistent JSON error the model can act on. Keeping them
here stops each tool from inventing its own.
"""

import json

import pandas as pd

from src.services import CompanyNotFoundError, DataUnavailableError, list_companies, load_company_df


class BadDateError(ValueError):
    """A date argument was not written as YYYY-MM-DD."""


class EmptyRangeError(ValueError):
    """The company has no recorded trading days inside the requested range."""


def tool_error(message: str, **extra) -> str:
    """
    Render a failure as JSON for the model.

    Tools report failures rather than raising them, so that a bad argument sends
    the agent back with something it can correct instead of ending the turn.
    """
    return json.dumps({"error": message, **extra})


def company_not_found_error(company: str) -> str:
    """Report an unknown symbol, listing the ones that do exist."""
    try:
        available = ", ".join(sorted(list_companies()))
    except Exception:
        available = "unavailable"
    return tool_error(f"No data for '{company}'.", available_companies=available)


def load_window(
    company: str, start_date: str = None, end_date: str = None
) -> pd.DataFrame:
    """
    Load a company's price history, narrowed to a date range and sorted oldest first.

    Args:
        company: the ticker symbol.
        start_date: optional inclusive start, as YYYY-MM-DD.
        end_date: optional inclusive end, as YYYY-MM-DD.

    Raises:
        CompanyNotFoundError: the symbol is not tracked.
        DataUnavailableError: the dataset could not be fetched.
        BadDateError: a date argument could not be parsed.
        EmptyRangeError: no trading days fall inside the range.
    """
    df = load_company_df(company)

    try:
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
    except Exception as e:
        raise BadDateError("start_date and end_date must be formatted as YYYY-MM-DD.") from e

    if df.empty:
        raise EmptyRangeError(f"No rows recorded for '{company}' in that date range.")

    return df.sort_values("date")


def handle_frame_error(company: str, error: Exception) -> str:
    """
    Turn any failure from load_window into the JSON the model should see.

    Returns None when the error is not one this helper knows about, so the caller
    can decide what to do with it.
    """
    if isinstance(error, CompanyNotFoundError):
        return company_not_found_error(company)
    if isinstance(error, (BadDateError, EmptyRangeError)):
        return tool_error(str(error))
    if isinstance(error, DataUnavailableError):
        return tool_error(str(error))
    return None
