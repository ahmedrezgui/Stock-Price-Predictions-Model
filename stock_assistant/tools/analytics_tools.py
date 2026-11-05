"""
Analytics tools: the calculations the assistant is allowed to perform on prices.

These are named calculations rather than a general code-running tool. The
assistant is driven by whatever a user types into a public chat box, so handing it
an interpreter would hand the same to anyone who asks; a fixed set of functions
gives the same analytical reach without that.
"""

import json
from typing import List, Optional

import pandas as pd
from langchain_core.tools import tool

from market_data.history import list_companies

from .frames import handle_frame_error, load_window, tool_error
from .indicators import (
    annualised_volatility_pct,
    daily_returns,
    exponential_moving_average,
    last_value,
    max_drawdown_pct,
    relative_strength_index,
    simple_moving_average,
    total_return_pct,
)

# Comparing more names than this produces a table nobody reads and a correlation
# matrix that will not fit on screen.
MAX_COMPARED = 6

# How many gainers and losers top_movers reports on each side.
DEFAULT_MOVERS = 5


@tool
def analyze_stock_performance(
    company: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Measure how one Tunisian stock has performed over a period.

    Use this instead of working figures out yourself whenever the user asks how a
    stock has done, how risky or volatile it is, whether it is trending up or
    down, or where it sits against its moving averages.

    Args:
        company: the ticker symbol, e.g. "BIAT".
        start_date: optional start of the period, as YYYY-MM-DD. Defaults to the
            beginning of the record.
        end_date: optional end of the period, as YYYY-MM-DD. Defaults to the most
            recent day recorded.

    Returns:
        A JSON object with the total return, annualised volatility, worst
        drawdown, moving averages, RSI and average volume. A metric comes back as
        null when the period was too short to compute it honestly, in which case
        say so rather than filling it in.
    """
    try:
        df = load_window(company, start_date, end_date)
    except Exception as e:
        return handle_frame_error(company, e) or tool_error(f"Could not analyse '{company}': {e}")

    closes = df["cloture"]

    return json.dumps(
        {
            "company": company,
            "period": {
                "from": df["date"].iloc[0].strftime("%Y-%m-%d"),
                "to": df["date"].iloc[-1].strftime("%Y-%m-%d"),
                "trading_days": int(len(df)),
            },
            "performance": {
                "first_close": round(float(closes.iloc[0]), 3),
                "last_close": round(float(closes.iloc[-1]), 3),
                "total_return_pct": total_return_pct(closes),
                "annualised_volatility_pct": annualised_volatility_pct(closes),
                "max_drawdown_pct": max_drawdown_pct(closes),
                "average_daily_volume": round(float(df["volume"].mean()), 2),
            },
            "indicators": {
                "sma_20": last_value(simple_moving_average(closes, 20), 20),
                "sma_50": last_value(simple_moving_average(closes, 50), 50),
                "ema_20": last_value(exponential_moving_average(closes, 20), 20),
                "rsi_14": last_value(relative_strength_index(closes), 15),
            },
            "notes": (
                "Volatility is annualised from daily returns over 252 trading days. "
                "RSI above 70 is conventionally read as overbought, below 30 as "
                "oversold. Null means the period was too short for that measure."
            ),
        }
    )


@tool
def compare_stocks(
    companies: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Compare the performance of several Tunisian stocks over the same period.

    Use this when the user asks which of some stocks did better, or how they move
    in relation to one another.

    Args:
        companies: two to six ticker symbols.
        start_date: optional start of the period, as YYYY-MM-DD.
        end_date: optional end of the period, as YYYY-MM-DD.

    Returns:
        A JSON object with per-company returns and volatility, and a correlation
        matrix of their daily returns. Correlation near 1 means they move
        together, near 0 that they move independently, and negative that they
        move in opposite directions; it is computed only over the days on which
        every company traded.
    """
    if not companies:
        return tool_error("Name at least two companies to compare.")
    if len(companies) > MAX_COMPARED:
        return tool_error(f"Compare at most {MAX_COMPARED} companies at once; {len(companies)} were given.")

    summaries = {}
    return_series = {}
    failures = {}

    for company in companies:
        try:
            df = load_window(company, start_date, end_date)
        except Exception as e:
            failures[company] = handle_frame_error(company, e) or str(e)
            continue

        closes = df["cloture"]
        summaries[company] = {
            "first_close": round(float(closes.iloc[0]), 3),
            "last_close": round(float(closes.iloc[-1]), 3),
            "total_return_pct": total_return_pct(closes),
            "annualised_volatility_pct": annualised_volatility_pct(closes),
            "max_drawdown_pct": max_drawdown_pct(closes),
            "trading_days": int(len(df)),
        }
        # Indexed by date so that companies with different trading calendars line
        # up correctly rather than by row position.
        return_series[company] = daily_returns(closes).set_axis(df["date"].iloc[1:])

    if not summaries:
        return tool_error("None of those companies could be loaded.", details=failures)

    payload = {"companies": summaries}

    if len(return_series) > 1:
        aligned = pd.DataFrame(return_series).dropna()
        if len(aligned) > 1:
            correlation = aligned.corr().round(3)
            payload["correlation"] = {row: correlation.loc[row].to_dict() for row in correlation.index}
            payload["correlation_days"] = int(len(aligned))
        else:
            payload["correlation"] = None
            payload["correlation_note"] = "Too few days on which all of them traded to correlate."

    if failures:
        payload["unavailable"] = failures

    return json.dumps(payload)


@tool
def top_movers(period_days: int = 30, limit: int = DEFAULT_MOVERS) -> str:
    """Find the Tunisian stocks that rose and fell the most over a recent period.

    Use this for questions about what is moving in the market, the best or worst
    performers, or where to look first. It reads every tracked company, so prefer
    the other tools when the user has already named one.

    Args:
        period_days: how many calendar days back to measure from the most recent
            recorded day (default 30).
        limit: how many names to report on each side (default 5).

    Returns:
        A JSON object with the biggest gainers and the biggest losers, each with
        its percentage change over the period.
    """
    if period_days < 2:
        return tool_error("period_days must be at least 2.")

    try:
        companies = list_companies()
    except Exception as e:
        return tool_error(f"Could not list companies: {e}")

    performances = []
    skipped = []

    for company in companies:
        try:
            df = load_window(company)
        except Exception:
            skipped.append(company)
            continue

        # Measured back from each company's own last trading day, so a stock that
        # stopped trading early is not silently compared over a different window.
        cutoff = df["date"].iloc[-1] - pd.Timedelta(days=period_days)
        window = df[df["date"] >= cutoff]
        if len(window) < 2:
            skipped.append(company)
            continue

        performances.append(
            {
                "company": company,
                "change_pct": total_return_pct(window["cloture"]),
                "last_close": round(float(window["cloture"].iloc[-1]), 3),
                "from": window["date"].iloc[0].strftime("%Y-%m-%d"),
                "to": window["date"].iloc[-1].strftime("%Y-%m-%d"),
            }
        )

    if not performances:
        return tool_error("No company had enough recent data to rank.")

    ranked = sorted(
        [p for p in performances if p["change_pct"] is not None],
        key=lambda p: p["change_pct"],
        reverse=True,
    )

    payload = {
        "period_days": period_days,
        "companies_ranked": len(ranked),
        "gainers": ranked[:limit],
        "losers": list(reversed(ranked[-limit:])) if len(ranked) > limit else [],
    }
    if skipped:
        payload["skipped"] = skipped

    return json.dumps(payload)
