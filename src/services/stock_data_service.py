"""
Shared access to the DVC-tracked stock datasets and the trained forecasting models.

Both the web API and the RAG agent tools read stock data through this module, so it
raises plain domain errors instead of HTTP ones. Callers translate them as needed.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import predict
from ..handlers import get_model, get_scaler


DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
)


class StockDataError(Exception):
    """Base class for stock data failures."""


class CompanyNotFoundError(StockDataError):
    """No dataset is tracked for the requested company."""


class DataUnavailableError(StockDataError):
    """The dataset is tracked but could not be fetched or read."""


class ModelUnavailableError(StockDataError):
    """No usable model/scaler pair for the requested company."""


def _model_location() -> str:
    return os.environ.get("MODEL_LOCATION")


def _window_size() -> int:
    return int(os.environ.get("WINDOW_SIZE"))


def list_companies() -> list[str]:
    """Return the symbols of every company with a processed dataset."""
    return [
        f.replace(".csv.dvc", "")
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv.dvc")
    ]


def load_company_df(company: str) -> pd.DataFrame:
    """
    Load a company's processed OHLCV history, pulling it from remote storage if the
    CSV is not present locally.

    Raises:
        CompanyNotFoundError: the company is not tracked.
        DataUnavailableError: the dataset could not be pulled.
    """
    dvc_file = os.path.join(DATA_DIR, f"{company}.csv.dvc")
    if not os.path.exists(dvc_file):
        raise CompanyNotFoundError("Company not found")

    csv_file = dvc_file.replace(".dvc", "")
    if not os.path.exists(csv_file):
        try:
            subprocess.run(
                ["dvc", "pull", dvc_file], check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            raise DataUnavailableError(
                f"Failed to pull data with DVC. stdout: {e.stdout.strip() if e.stdout else ''}, "
                f"stderr: {e.stderr.strip() if e.stderr else ''}"
            ) from e

    df = pd.read_csv(csv_file, sep=";")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", dayfirst=True)
    return df


def _forecast_from_df(company: str, df: pd.DataFrame, horizon: int = None) -> dict:
    window_size = _window_size()
    horizon = horizon or window_size
    model_location = _model_location()

    model = get_model(company, model_location)
    scaler = get_scaler(company, model_location)
    if model is None or scaler is None:
        raise ModelUnavailableError("Model unavailable, Try later")

    last_date = max(df["date"])
    # A model trained on anything other than the latest data would extrapolate from a
    # different series than the one we are about to return.
    if model.get_last_trained_date() != last_date:
        raise ModelUnavailableError("Model unavailable, Try later")

    values = predict(
        model, df["cloture"].values.reshape(-1, 1), scaler, horizon, window_size
    )
    dates = [last_date + relativedelta(days=i + 1) for i in range(len(values))]

    return {
        "company": company,
        "last_date": last_date,
        "last_close": float(df["cloture"].iloc[-1]),
        "dates": dates,
        "values": [float(v) for v in values],
    }


def get_prediction(company: str, horizon: int = None) -> dict:
    """
    Forecast a company's closing price.

    Args:
        company: stock symbol, e.g. "BIAT".
        horizon: number of days to forecast; defaults to the model's window size.

    Returns:
        dict with the company, its last known close and date, and the forecast
        dates/values.

    Raises:
        CompanyNotFoundError, DataUnavailableError, ModelUnavailableError.
    """
    df = load_company_df(company)
    return _forecast_from_df(company, df, horizon)


def get_stock_with_forecast(company: str) -> dict:
    """
    Return a company's full price history with the forecast appended as extra rows.

    Raises:
        CompanyNotFoundError, DataUnavailableError, ModelUnavailableError.
    """
    df = load_company_df(company)
    forecast = _forecast_from_df(company, df)

    for date, val in zip(forecast["dates"], forecast["values"]):
        # date;ouverture;haut;bas;cloture;volume
        df.loc[len(df)] = [date, val, val, val, val, 0]

    return {"columns": df.columns.tolist(), "data": df.to_dict(orient="records")}
