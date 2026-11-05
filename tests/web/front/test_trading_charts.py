import numpy as np
import pandas as pd
import pytest

from apps.front.trading_charts import build_chart, indicators


def history(size=260):
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=size), "cloture": np.arange(1, size + 1, dtype=float), "volume": np.arange(size) * 100})


def test_moving_averages_have_warmup_and_correct_values():
    result = indicators(history(), (20, 50), (20,))
    assert result["SMA 20"].iloc[:19].isna().all()
    assert result["SMA 20"].iloc[19] == 10.5
    assert result["SMA 50"].iloc[49] == 25.5
    assert result["EMA 20"].iloc[-1] == pytest.approx(history().cloture.ewm(span=20, adjust=False).mean().iloc[-1])
    assert result["RSI 14"].iloc[-1] == 100


def test_flat_series_rsi_and_bands():
    frame = history()
    frame["cloture"] = 10.0
    result = indicators(frame)
    assert result["RSI 14"].iloc[-1] == 50
    assert result["BB upper"].iloc[-1] == result["BB lower"].iloc[-1] == 10
    assert result.Histogram.iloc[-1] == 0


def test_indicators_do_not_use_future_data():
    frame = history()
    expected = indicators(frame.iloc[:200])
    frame.loc[200:, "cloture"] = 99999
    actual = indicators(frame).iloc[:200]
    pd.testing.assert_frame_equal(expected, actual)


def test_chart_uses_full_warmup_and_separate_volume_axis():
    chart = build_chart(history(), "AL", style="Line", sma_windows=(200,), momentum="MACD", period="1M", bands=True)
    traces = {trace.name: trace for trace in chart.data}
    assert not np.isnan(traces["SMA 200"].y[0])
    assert traces["Volume"].yaxis != traces["Close"].yaxis
    assert traces["MACD"].yaxis != traces["Volume"].yaxis
    assert "BB upper" in traces


def test_short_history_no_invented_indicators():
    result = indicators(history(3))
    assert result["RSI 14"].isna().all()
    assert result["SMA 20"].isna().all()


def test_monthly_ohlcv_aggregation_and_year_axis():
    from apps.front.trading_charts import aggregate_bars
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-31", "2024-02-01"]),
        "ouverture": [9., 10., 12., 14.], "haut": [11., 13., 15., 16.],
        "bas": [8., 9., 11., 13.], "cloture": [10., 12., 14., 15.],
        "volume": [100., 200., 300., 400.],
    })
    monthly = aggregate_bars(frame, "Month")
    january = monthly.iloc[1]
    assert (january.ouverture, january.haut, january.bas, january.cloture, january.volume) == (10, 15, 9, 14, 500)
    chart = build_chart(frame, "BIAT", interval="Year", period="All", sma_windows=())
    assert list(chart.data[0].x) == ["2023", "2024"]
    assert list(chart.data[0].close) == [10, 15]
    assert chart.layout.xaxis.type == "category"
    assert chart.data[0].increasing.fillcolor
    frame.loc[1, "haut"] = np.nan
    frame.loc[1, "volume"] = np.nan
    missing = aggregate_bars(frame, "Month").iloc[1]
    assert np.isnan(missing.haut) and np.isnan(missing.volume)
    assert missing.cloture == 14


def test_monthly_indicators_use_months_and_keep_warmup():
    frame = history(900)
    chart = build_chart(frame, "BIAT", interval="Month", period="3M", sma_windows=(20,))
    from apps.front.trading_charts import aggregate_bars
    expected = indicators(aggregate_bars(frame, "Month"), (20,))
    trace = next(t for t in chart.data if t.name == "SMA 20")
    assert trace.y[-1] == pytest.approx(expected["SMA 20"].iloc[-1])
    assert len(trace.x) <= 4
