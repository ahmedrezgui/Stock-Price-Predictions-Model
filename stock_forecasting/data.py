"""Daily bars: no filling, explicit adjustments, causal features and label purging."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config

FEATURE_VERSION = "causal-v1"
ALIASES = {
    "date": "timestamp",
    "symbole": "symbol",
    "ouverture": "open",
    "haut": "high",
    "bas": "low",
    "cloture": "close",
    "adj close": "adjusted_close",
}


class Provider(Protocol):
    def load(self, symbol: str) -> pd.DataFrame: ...


class CSVProvider:
    """Adapter for existing free ilboursa downloads or canonical OHLCV files."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self, symbol: str) -> pd.DataFrame:
        return read_csv(self.root / f"{symbol}.csv")


def read_csv(path: str | Path) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8-sig") as stream:
        first = stream.readline()
    return pd.read_csv(path, sep=";" if ";" in first else ",", encoding="utf-8-sig")


def sessions(config: Config, start, end) -> pd.DatetimeIndex | None:
    if config.session_file:
        values = pd.read_csv(config.session_file)["session"]
        index = pd.DatetimeIndex(pd.to_datetime(values)).normalize().sort_values().unique()
        if index.min() > pd.Timestamp(start) or index.max() < pd.Timestamp(end):
            raise ValueError("Session file does not cover requested range")
        return index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    if config.calendar:
        import exchange_calendars as xcals

        cal = xcals.get_calendar(config.calendar, start=str(start), end=str(end))
        index = cal.sessions_in_range(start, end)
        return index.tz_localize(None) if index.tz is not None else index
    return None


def forecast_session(config: Config, as_of: pd.Timestamp) -> str | None:
    """Use only an explicitly configured calendar; never guess with weekdays."""
    if config.session_file:
        index = (
            pd.DatetimeIndex(pd.to_datetime(pd.read_csv(config.session_file)["session"]))
            .normalize()
            .sort_values()
            .unique()
        )
    elif config.calendar:
        index = sessions(config, as_of, as_of + pd.Timedelta(days=config.horizon * 4 + 30))
    else:
        return None
    future = index[index > as_of]
    if len(future) < config.horizon:
        raise ValueError("Calendar does not cover forecast horizon")
    return str(future[config.horizon - 1].date())


def normalize(raw: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict]:
    frame = raw.copy()
    frame.columns = [ALIASES.get(str(c).lower().strip(), str(c).lower().strip()) for c in frame.columns]
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if not set(required) <= set(frame):
        raise ValueError(f"Missing columns: {sorted(set(required) - set(frame))}")
    if "symbol" in frame and set(frame.symbol.dropna().astype(str)) != {config.symbol}:
        raise ValueError("Dataset must contain exactly the configured symbol")
    frame["symbol"] = config.symbol
    date_strings = frame.timestamp.astype(str)
    frame["timestamp"] = pd.to_datetime(date_strings, dayfirst=date_strings.str.contains("/").any()).dt.normalize()
    if frame.timestamp.isna().any() or frame.timestamp.dt.tz is not None:
        raise ValueError("Use timezone-free exchange session dates, not intraday timestamps")
    for col in required[1:] + (["adjusted_close"] if "adjusted_close" in frame else []):
        frame[col] = pd.to_numeric(frame[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    frame = frame.sort_values("timestamp").drop_duplicates()
    if frame.timestamp.duplicated().any():
        raise ValueError("Conflicting duplicate sessions; reconcile at the provider")
    n_original = len(frame)
    # Without an exchange close timestamp, exclude the current date conservatively.
    cutoff = pd.Timestamp(config.as_of) if config.as_of else pd.Timestamp.now(tz="UTC").tz_localize(None)
    frame = frame[frame.timestamp < cutoff.normalize()]
    if config.start:
        frame = frame[frame.timestamp >= pd.Timestamp(config.start)]
    if config.end:
        frame = frame[frame.timestamp <= pd.Timestamp(config.end)]
    if frame.empty:
        raise ValueError("No completed observations in requested date range")
    prices = frame[["open", "high", "low", "close"]]
    if not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError("OHLC prices must be finite and strictly positive; no filling is allowed")
    if ((frame.high < prices.max(axis=1)) | (frame.low > prices.min(axis=1))).any():
        raise ValueError("OHLC range inconsistency")
    if frame.volume.isna().any() or not np.isfinite(frame.volume).all() or (frame.volume < 0).any():
        raise ValueError("Missing/invalid volume; provider reconciliation required")
    frame["raw_close"] = frame.close
    if config.adjustment == "adjusted":
        if (
            "adjusted_close" not in frame
            or not np.isfinite(frame.adjusted_close).all()
            or (frame.adjusted_close <= 0).any()
        ):
            raise ValueError("Adjusted policy requires verified positive adjusted_close")
        ratio = frame.adjusted_close / frame.close
        frame[["open", "high", "low", "close"]] = prices.mul(ratio, axis=0)
    index = sessions(config, frame.timestamp.min(), frame.timestamp.max())
    if index is not None:
        if not frame.timestamp.isin(index).all():
            raise ValueError("Observations fall outside the specified exchange calendar")
        missing = index.difference(pd.DatetimeIndex(frame.timestamp))
        if len(missing):
            raise ValueError(f"Missing {len(missing)} exchange sessions; cannot claim session horizons")
    frame["source"] = config.source
    frame = frame.reset_index(drop=True)
    metadata = {
        "symbol": config.symbol,
        "source": config.source,
        "rows": len(frame),
        "excluded_by_date": n_original - len(frame),
        "start": str(frame.timestamp.iloc[0].date()),
        "end": str(frame.timestamp.iloc[-1].date()),
        "zero_volume_rows": int((frame.volume == 0).sum()),
        "adjustment": config.adjustment,
        "calendar": config.calendar,
        "horizon_basis": "exchange_sessions" if index is not None else "observed_bars_unverified_sessions",
        "timezone_convention": "exchange-local session date; current UTC date excluded",
        "corporate_actions": "provider adjustment required; no inferred split corrections",
    }
    return frame, metadata


def features(frame: pd.DataFrame, groups: list[str]) -> tuple[pd.DataFrame, list[str]]:
    f = frame.copy()
    p, v = f.close, f.volume
    r = np.log(p / p.shift(1))
    f["log_return"] = r
    columns = ["log_return"]
    additions = {}
    if "price" in groups:
        additions.update({f"return_lag_{lag}": r.shift(lag) for lag in (1, 2, 5)})
        for w in (5, 20):
            additions[f"momentum_{w}"] = np.log(p / p.shift(w))
            additions[f"ma_distance_{w}"] = p / p.rolling(w).mean() - 1
        additions.update(
            intraday_range=np.log(f.high / f.low),
            open_close=np.log(p / f.open),
            overnight_gap=np.log(f.open / p.shift(1)),
        )
    if "volatility" in groups:
        tr = pd.concat([f.high - f.low, abs(f.high - p.shift(1)), abs(f.low - p.shift(1))], axis=1).max(axis=1)
        additions.update(
            volatility_20=r.rolling(20).std(),
            ewm_volatility=r.ewm(span=20, adjust=False).std(),
            atr_14=tr.rolling(14).mean() / p,
            high_low_volatility=np.log(f.high / f.low).rolling(20).std(),
            parkinson_20=np.sqrt((np.log(f.high / f.low) ** 2).rolling(20).mean() / (4 * np.log(2))),
        )
    if "volume" in groups:
        vm, vs = v.rolling(20).mean(), v.rolling(20).std()
        additions.update(
            relative_volume=v / vm.replace(0, np.nan),
            volume_change=np.log1p(v).diff(),
            log_mean_volume=np.log1p(vm),
            volume_zscore=(v - vm) / vs.replace(0, np.nan),
            log_dollar_volume=np.log1p(f.raw_close * v),
        )
    if "technical" in groups:
        delta = p.diff()
        up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rsi = (up / (up + down)).fillna(0.5)
        macd = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
        additions.update(
            rsi_14=rsi,
            macd=macd / p,
            macd_signal=macd.ewm(span=9, adjust=False).mean() / p,
            bollinger_z=(p - p.rolling(20).mean()) / p.rolling(20).std().replace(0, np.nan),
        )
    for name, values in additions.items():
        f[name] = values
        columns.append(name)
    if np.isinf(f[columns].to_numpy()).any():
        raise ValueError("Infinite features")
    return f, columns


def available_context(frame: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of join on true release timestamp, never observation period."""
    if "available_at" not in context:
        raise ValueError("External features require available_at publication timestamps")
    return pd.merge_asof(
        frame.sort_values("timestamp"),
        context.sort_values("available_at"),
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
    )


def targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("Positive horizon required")
    f = frame.copy()
    f["target_return"] = np.log(f.close.shift(-horizon) / f.close)
    f["target_direction"] = (f.target_return > 0).astype(float).where(f.target_return.notna())
    f["target_timestamp"] = f.timestamp.shift(-horizon)
    return f


@dataclass
class Samples:
    x: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    ends: np.ndarray


def sequences(frame: pd.DataFrame, columns: list[str], length: int, horizon: int) -> Samples:
    if any(c.startswith("target_") for c in columns):
        raise ValueError("Future target columns cannot be features")
    values = frame[columns].to_numpy(dtype=np.float64)
    origins = np.array(
        [
            i
            for i in range(length - 1, len(frame) - horizon)
            if np.isfinite(values[i - length + 1 : i + 1]).all() and np.isfinite(frame.target_return.iloc[i])
        ],
        dtype=int,
    )
    if not len(origins):
        raise ValueError("Insufficient finite history for sequences")
    return Samples(
        np.stack([values[i - length + 1 : i + 1] for i in origins]),
        frame.target_return.iloc[origins].to_numpy(),
        origins,
        origins + horizon,
    )


def split_masks(samples: Samples, train_end: int, validation_end: int, test_end: int | None = None):
    # Strict inequality: every training label is known BEFORE the first validation origin.
    train = samples.ends < train_end
    validation = (samples.origins >= train_end) & (samples.ends < validation_end)
    test = samples.origins >= validation_end
    if test_end is not None:
        test &= samples.ends < test_end
    if not all(np.any(m) for m in (train, validation, test)):
        raise ValueError("Split too small after sequence warmup and horizon purging")
    return train, validation, test


def fit_scaler(frame: pd.DataFrame, columns: list[str], samples: Samples, mask: np.ndarray, length: int):
    # Fit each legitimately used training input row once, never future label rows.
    rows = np.unique(np.concatenate([np.arange(i - length + 1, i + 1) for i in samples.origins[mask]]))
    return StandardScaler().fit(frame.iloc[rows][columns]), rows


def transform(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return ((x - scaler.mean_) / scaler.scale_).astype(np.float32)
