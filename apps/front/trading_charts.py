"""Display indicators computed from completed source bars, never forecast inputs."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ["#f5b85a", "#8c9eff", "#ed88c8", "#58c6ed", "#b7db78", "#ff896f"]
UP, DOWN = "#32d6b3", "#f36c83"


def aggregate_bars(frame, interval="Day"):
    """Group observed bars without inventing OHLC or missing volume values."""
    data = frame.sort_values("date").copy()
    if interval == "Day":
        return data
    frequencies = {"Month": "M", "Year": "Y"}
    if interval not in frequencies:
        raise ValueError("Interval must be Day, Month or Year")
    rows = []
    for _, group in data.groupby(data.date.dt.to_period(frequencies[interval])):
        row = {"date": group.date.iloc[0], "cloture": group.cloture.iloc[-1]}
        if {"ouverture", "haut", "bas"}.issubset(group):
            valid = group[["ouverture", "haut", "bas", "cloture"]].notna().all().all()
            row.update(ouverture=group.ouverture.iloc[0] if valid else np.nan,
                       haut=group.haut.max() if valid else np.nan,
                       bas=group.bas.min() if valid else np.nan)
        if "volume" in group:
            row["volume"] = group.volume.sum(min_count=len(group))
        rows.append(row)
    return pd.DataFrame(rows, columns=data.columns)


def indicators(frame, sma_windows=(20, 50), ema_windows=()):
    result = frame.sort_values("date").copy()
    close = result.cloture
    for window in sma_windows:
        result[f"SMA {window}"] = close.rolling(window, min_periods=window).mean()
    for window in ema_windows:
        result[f"EMA {window}"] = close.ewm(span=window, adjust=False, min_periods=window).mean()
    mean, std = close.rolling(20).mean(), close.rolling(20).std(ddof=0)
    result["BB upper"], result["BB lower"] = mean + 2 * std, mean - 2 * std
    delta = close.diff()
    # Wilder smoothing starts with the first 14-observation simple average.
    def wilder(values):
        seeded = pd.Series(np.nan, index=values.index)
        if len(values) > 14:
            seeded.iloc[14] = values.iloc[1:15].mean()
            seeded.iloc[15:] = values.iloc[15:]
        return seeded.ewm(alpha=1 / 14, adjust=False, ignore_na=True).mean()
    gain, loss = wilder(delta.clip(lower=0)), wilder(-delta.clip(upper=0))
    result["RSI 14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    result.loc[(loss == 0) & (gain > 0), "RSI 14"] = 100
    result.loc[(gain == 0) & (loss > 0), "RSI 14"] = 0
    result.loc[(gain == 0) & (loss == 0), "RSI 14"] = 50
    result["MACD"] = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["Signal"] = result.MACD.ewm(span=9, adjust=False, min_periods=9).mean()
    result["Histogram"] = result.MACD - result.Signal
    result["Volume color"] = np.where(close.diff().ge(0), UP, DOWN)
    return result


def build_chart(frame, symbol, style="Candles", sma_windows=(20, 50), ema_windows=(), bands=False, momentum="RSI", period="1Y", volume=True, interval="Day"):
    # Calculate first, slice second: changing the viewport must not reset indicators.
    data = indicators(aggregate_bars(frame, interval), sma_windows, ema_windows)
    offsets = {"1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1), "5Y": pd.DateOffset(years=5)}
    if period in offsets:
        cutoff = frame.date.max() - offsets[period]
        if interval != "Day":
            cutoff = cutoff.to_period("M" if interval == "Month" else "Y").start_time
        data = data[data.date >= cutoff]
    # Equally spaced observed bars remove empty weekends and long market gaps.
    # String categories also give monthly/yearly candles a readable body width.
    data = data.copy()
    data["date"] = data.date.dt.strftime({"Day": "%Y-%m-%d", "Month": "%Y-%m", "Year": "%Y"}[interval])
    panels = ["Price"] + (["Volume"] if volume else []) + ([momentum] if momentum != "None" else [])
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.65] + [0.175] * (len(panels) - 1))
    candles = {"ouverture", "haut", "bas"}.issubset(data.columns)
    if style == "Candles" and candles:
        fig.add_trace(go.Candlestick(x=data.date, open=data.ouverture, high=data.haut, low=data.bas, close=data.cloture, name=symbol, increasing=dict(line=dict(color=UP, width=1.5), fillcolor=UP), decreasing=dict(line=dict(color=DOWN, width=1.5), fillcolor=DOWN), whiskerwidth=0.4), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=data.date, y=data.cloture, name="Close", mode="lines", line=dict(color=UP, width=2)), row=1, col=1)
    overlays = [f"SMA {w}" for w in sma_windows] + [f"EMA {w}" for w in ema_windows]
    for index, name in enumerate(overlays):
        fig.add_trace(go.Scatter(x=data.date, y=data[name], name=name, mode="lines", line=dict(color=COLORS[index % len(COLORS)], width=1.4, dash="dot" if name.startswith("EMA") else "solid")), row=1, col=1)
    if bands:
        for name, fill in [("BB upper", None), ("BB lower", "tonexty")]:
            fig.add_trace(go.Scatter(x=data.date, y=data[name], name=name, mode="lines", line=dict(color="#7895af", width=1), fill=fill, fillcolor="rgba(120,149,175,0.08)"), row=1, col=1)
    if volume:
        values = data.get("volume", pd.Series(np.nan, index=data.index))
        fig.add_trace(go.Bar(x=data.date, y=values, name="Volume", marker_color=data["Volume color"], opacity=0.65), row=panels.index("Volume") + 1, col=1)
    if momentum == "RSI":
        row = panels.index("RSI") + 1
        fig.add_trace(go.Scatter(x=data.date, y=data["RSI 14"], name="RSI 14", line=dict(color="#b49bff", width=1.5)), row=row, col=1)
        for value in [30, 70]:
            fig.add_hline(y=value, line_dash="dot", line_color="#526077", row=row, col=1)
        fig.update_yaxes(range=[0, 100], row=row, col=1)
    elif momentum == "MACD":
        row = panels.index("MACD") + 1
        fig.add_trace(go.Bar(x=data.date, y=data.Histogram, name="MACD histogram", marker_color=np.where(data.Histogram.ge(0), UP, DOWN)), row=row, col=1)
        for name, color in [("MACD", "#58c6ed"), ("Signal", "#f5b85a")]:
            fig.add_trace(go.Scatter(x=data.date, y=data[name], name=name, line=dict(color=color, width=1.4)), row=row, col=1)
    for row, name in enumerate(panels, 1):
        fig.update_yaxes(title_text=name, side="right", row=row, col=1)
    fig.update_layout(height=650 if len(panels) > 1 else 500, paper_bgcolor="#101722", plot_bgcolor="#101722", font=dict(color="#b7c4d6", size=11), margin=dict(l=12, r=18, t=45, b=20), hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0), dragmode="pan", uirevision=f"{symbol}-{period}-{style}-{momentum}-{volume}")
    fig.update_layout(height=760 if len(panels) > 1 else 600, uirevision=f"{symbol}-{period}-{style}-{momentum}-{volume}-{interval}", bargap=0.25)
    step = max(1, int(np.ceil(len(data) / 9)))
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=data.date.tolist(), tickmode="array", tickvals=data.date.iloc[::step].tolist(), rangeslider_visible=False, gridcolor="#202b3b", showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(gridcolor="#202b3b", zeroline=False, fixedrange=False)
    return fig
