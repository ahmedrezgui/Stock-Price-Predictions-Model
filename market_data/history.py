"""Read validated local bars; the assistant never downloads or changes datasets."""

import os
from pathlib import Path
from stock_forecasting.config import Config
from stock_forecasting.data import normalize, read_csv
from .stock_data_service import CompanyNotFoundError, DataUnavailableError


def data_root():
    return Path(os.getenv("ASSISTANT_DATA_ROOT", os.getenv("FORECAST_DATA_ROOT", "data/raw")))


def list_companies():
    return sorted(path.stem for path in data_root().glob("*.csv"))


def load_company_df(company):
    company = company.upper()
    try:
        config = Config(symbol=company)
    except ValueError:
        raise CompanyNotFoundError("Invalid stock symbol") from None
    path = data_root() / f"{company}.csv"
    if not path.is_file():
        raise CompanyNotFoundError(f"No local dataset for {company}")
    try:
        frame, _ = normalize(read_csv(path), config)
    except (ValueError, OSError) as exc:
        raise DataUnavailableError(f"Cannot read validated history for {company}: {exc}") from None
    return frame.rename(
        columns={"timestamp": "date", "open": "ouverture", "high": "haut", "low": "bas", "close": "cloture"}
    )
