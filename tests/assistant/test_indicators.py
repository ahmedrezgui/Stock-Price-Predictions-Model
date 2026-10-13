import math

import pandas as pd
import pytest

from stock_assistant.tools.indicators import (
    annualised_volatility_pct,
    daily_returns,
    exponential_moving_average,
    last_value,
    max_drawdown_pct,
    relative_strength_index,
    simple_moving_average,
    total_return_pct,
)


def series(*values):
    return pd.Series([float(v) for v in values])


def test_daily_returns_drops_the_undefined_first_day():
    returns = daily_returns(series(100, 110, 99))

    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)


def test_total_return_is_first_to_last():
    assert total_return_pct(series(100, 50, 125)) == 25.0


def test_total_return_of_a_flat_series_is_zero():
    assert total_return_pct(series(100, 100, 100)) == 0.0


def test_total_return_guards_against_a_zero_start():
    assert total_return_pct(series(0, 100)) is None


def test_volatility_is_annualised_from_daily_returns():
    closes = series(100, 110, 99, 108.9)
    expected = float(daily_returns(closes).std()) * math.sqrt(252) * 100

    assert annualised_volatility_pct(closes) == pytest.approx(expected, abs=0.01)


def test_volatility_needs_two_returns():
    assert annualised_volatility_pct(series(100, 110)) is None
    assert annualised_volatility_pct(series(100)) is None


def test_max_drawdown_measures_peak_to_trough():
    # Peaks at 120, falls to 60: a 50% drawdown, even though it recovers after.
    assert max_drawdown_pct(series(100, 120, 60, 130)) == -50.0


def test_max_drawdown_of_a_rising_series_is_zero():
    assert max_drawdown_pct(series(100, 110, 120)) == 0.0


def test_simple_moving_average_waits_for_a_full_window():
    average = simple_moving_average(series(1, 2, 3, 4), 3)

    assert pd.isna(average.iloc[0])
    assert pd.isna(average.iloc[1])
    assert average.iloc[2] == pytest.approx(2.0)
    assert average.iloc[3] == pytest.approx(3.0)


def test_exponential_moving_average_starts_at_the_first_value():
    average = exponential_moving_average(series(10, 20), span=2)

    assert average.iloc[0] == pytest.approx(10.0)
    # Weighted towards the newer value, so it sits above the simple mean of 15.
    assert average.iloc[1] > 15.0


def test_rsi_is_100_when_nothing_ever_falls():
    rsi = relative_strength_index(series(*range(100, 130)))

    assert rsi.dropna().iloc[-1] == pytest.approx(100.0)


def test_rsi_is_low_when_everything_falls():
    rsi = relative_strength_index(series(*range(130, 100, -1)))

    assert rsi.dropna().iloc[-1] < 5


def test_rsi_of_an_even_market_sits_near_the_middle():
    alternating = series(*[100 + (i % 2) for i in range(40)])

    assert 40 < relative_strength_index(alternating).dropna().iloc[-1] < 60


def test_last_value_rounds_and_ignores_the_leading_gap():
    assert last_value(simple_moving_average(series(1, 2, 3, 4), 3), 3) == 3.0


def test_last_value_is_none_when_the_series_is_too_short():
    assert last_value(simple_moving_average(series(1, 2), 50), 50) is None
    assert last_value(pd.Series(dtype=float)) is None
