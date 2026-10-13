"""
The technical indicator maths behind the analytics and plotting tools.

These are plain pandas functions with no knowledge of tools or agents, so they can
be read and tested on their own. Each returns None rather than a misleading number
when the series is too short to support it: a 50-day average of 12 days of trading
would be worse than no answer at all.
"""

import math

import pandas as pd

# Tunis trades roughly 252 days a year, the usual figure for annualising a daily
# standard deviation.
TRADING_DAYS_PER_YEAR = 252

RSI_PERIOD = 14


def daily_returns(closes: pd.Series) -> pd.Series:
    """Day-on-day fractional change, with the first (undefined) day dropped."""
    return closes.pct_change().dropna()


def total_return_pct(closes: pd.Series) -> float:
    """Percentage change from the first close in the window to the last."""
    first, last = float(closes.iloc[0]), float(closes.iloc[-1])
    if first == 0:
        return None
    return round((last - first) / first * 100, 2)


def annualised_volatility_pct(closes: pd.Series) -> float:
    """
    Annualised standard deviation of daily returns, as a percentage.

    Returns None below two returns, where a standard deviation is undefined.
    """
    returns = daily_returns(closes)
    if len(returns) < 2:
        return None
    return round(float(returns.std()) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100, 2)


def max_drawdown_pct(closes: pd.Series) -> float:
    """
    The deepest peak-to-trough fall within the window, as a negative percentage.

    This is the loss an investor who bought at the worst moment would have lived
    through, which volatility alone does not convey.
    """
    if closes.empty:
        return None
    running_peak = closes.cummax()
    drawdowns = (closes - running_peak) / running_peak
    return round(float(drawdowns.min()) * 100, 2)


def simple_moving_average(closes: pd.Series, window: int) -> pd.Series:
    """Rolling mean over the given number of trading days."""
    return closes.rolling(window=window).mean()


def exponential_moving_average(closes: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted mean, which reacts faster than the simple average."""
    return closes.ewm(span=span, adjust=False).mean()


def relative_strength_index(closes: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Wilder's RSI: where recent gains sit against recent losses, from 0 to 100.

    Above 70 is conventionally read as overbought and below 30 as oversold. This
    uses Wilder's smoothing (an EWM with alpha = 1/period) rather than a plain
    rolling mean, which is the definition most charting tools follow.
    """
    change = closes.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False).mean()

    # A window with no losing day has infinite relative strength, i.e. RSI 100.
    strength = average_gain / average_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + strength))
    return rsi.fillna(100).where(average_loss.notna())


def last_value(series: pd.Series, minimum_periods: int = 1) -> float:
    """
    The final value of an indicator, or None if there was not enough data for it.

    Args:
        series: the computed indicator.
        minimum_periods: how many observations the indicator needed to be valid.
    """
    if series is None or len(series.dropna()) < 1 or len(series) < minimum_periods:
        return None
    value = series.dropna().iloc[-1]
    return None if pd.isna(value) else round(float(value), 3)
