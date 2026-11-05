"""Read validated local bars; the assistant never downloads or changes datasets."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from stock_forecasting.config import Config
from stock_forecasting.data import ALIASES, normalize, read_csv
from .stock_data_service import CompanyNotFoundError, DataUnavailableError


def data_root():
    return Path(os.getenv("ASSISTANT_DATA_ROOT", os.getenv("FORECAST_DATA_ROOT", "data/raw")))


def list_companies():
    return sorted(path.stem for path in data_root().glob("*.csv"))


def load_closing_history(company, include_ohlcv=False):
    """Validate display-only closes without treating them as validated OHLC bars."""
    company = company.upper()
    try:
        Config(symbol=company)
    except ValueError:
        raise CompanyNotFoundError("Invalid stock symbol") from None
    path = data_root() / f"{company}.csv"
    if not path.is_file():
        raise CompanyNotFoundError(f"No local dataset for {company}")
    try:
        raw = read_csv(path)
        raw.columns = [ALIASES.get(str(c).lower().strip(), str(c).lower().strip()) for c in raw.columns]
        if "symbol" in raw and set(raw.symbol.dropna().astype(str)) != {company}:
            raise ValueError("Dataset symbol does not match")
        dates = raw["timestamp"].astype(str)
        frame = pd.DataFrame({
            "date": pd.to_datetime(dates, dayfirst=dates.str.contains("/").any()).dt.normalize(),
            "cloture": pd.to_numeric(raw["close"].astype(str).str.replace(",", ".", regex=False), errors="coerce"),
        })
        if frame.date.isna().any() or frame.date.dt.tz is not None:
            raise ValueError("Invalid session dates")
        frame = frame[frame.date < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()]
        frame = frame.sort_values("date").drop_duplicates()
        if frame.empty:
            raise ValueError("No completed observations")
        if frame.date.duplicated().any():
            raise ValueError("Conflicting closing prices for the same date")
        if not np.isfinite(frame.cloture).all() or (frame.cloture <= 0).any():
            raise ValueError("Closing prices must be finite and positive")
        if include_ohlcv:
            for source, target in [("open", "ouverture"), ("high", "haut"), ("low", "bas"), ("volume", "volume")]:
                values = pd.to_numeric(raw.get(source, pd.Series(index=raw.index, dtype=float)).astype(str).str.replace(",", ".", regex=False), errors="coerce")
                frame[target] = values.where(np.isfinite(values) & (values.ge(0) if source == "volume" else values.gt(0)))
            prices = frame[["ouverture", "haut", "bas", "cloture"]]
            valid = prices.notna().all(axis=1) & frame.haut.ge(prices.max(axis=1)) & frame.bas.le(prices.min(axis=1))
            frame.loc[~valid, ["ouverture", "haut", "bas"]] = np.nan
            frame.attrs["quality"] = {"invalid_ohlc_rows": int((~valid).sum()), "missing_volume_rows": int(frame.volume.isna().sum())}
        return frame.reset_index(drop=True)
    except (ValueError, KeyError, OSError) as exc:
        raise DataUnavailableError(f"Cannot read closing history for {company}: {exc}") from None


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
