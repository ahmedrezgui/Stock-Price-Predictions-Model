"""
Plotting tools: charts the assistant can draw for the user.

These tools differ from the rest in what they return. A figure is far too large to
put in front of the model — it would fill the context with coordinates it cannot
read anyway — so each returns a LangGraph Command that puts the figure into the
graph's state, where the API picks it up, and sends the model only a short
confirmation of what was drawn.
"""

from typing import Annotated, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from src.services import ModelUnavailableError, get_prediction

from .charts import MAX_SERIES, chart_payload, line_figure, series_color
from .frames import handle_frame_error, load_window, tool_error
from .indicators import (
    exponential_moving_average,
    simple_moving_average,
)

# What the user may ask to see drawn over the price line.
SUPPORTED_INDICATORS = {"sma_20", "sma_50", "ema_20"}

INDICATOR_LABELS = {
    "sma_20": "20-day average",
    "sma_50": "50-day average",
    "ema_20": "20-day EMA",
}


def _failed(tool_call_id: str, message: str) -> Command:
    """Report a failure through the same channel a drawn chart would use."""
    return Command(
        update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]}
    )


def _drawn(tool_call_id: str, chart: dict, summary: str) -> Command:
    """
    Hand the chart to the graph and tell the model only that it exists.

    The figure goes into state for the page to render; the model receives a
    sentence, which is all it needs to write an answer that refers to the chart.
    """
    return Command(
        update={
            "charts": [chart],
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


@tool
def plot_stock(
    company: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    indicators: Optional[List[str]] = None,
    include_forecast: bool = False,
) -> Command:
    """Draw the price history of one Tunisian stock, and optionally its forecast.

    Use this whenever a chart would say it better than a paragraph: a question
    about a trend, a movement over time, or an explicit request to see or plot
    something. The chart is shown to the user automatically, so write your answer
    as commentary on it rather than describing every point.

    Args:
        company: the ticker symbol, e.g. "BIAT".
        start_date: optional start of the period, as YYYY-MM-DD.
        end_date: optional end of the period, as YYYY-MM-DD.
        indicators: optional overlays, any of "sma_20", "sma_50", "ema_20".
        include_forecast: set true to extend the line with the model's forecast,
            drawn dashed to keep it distinct from recorded prices. Say in your
            answer that the dashed part is a prediction.

    Returns:
        Confirmation that the chart was drawn, or the reason it was not.
    """
    try:
        df = load_window(company, start_date, end_date)
    except Exception as e:
        return _failed(
            tool_call_id,
            handle_frame_error(company, e) or tool_error(f"Could not plot '{company}': {e}"),
        )

    dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
    closes = df["cloture"]
    series = [{"name": f"{company} close", "x": dates, "y": [float(c) for c in closes]}]

    unknown = sorted(set(indicators or []) - SUPPORTED_INDICATORS)
    if unknown:
        return _failed(
            tool_call_id,
            tool_error(
                f"Unsupported indicators: {', '.join(unknown)}.",
                supported=sorted(SUPPORTED_INDICATORS),
            ),
        )

    for name in indicators or []:
        values = (
            simple_moving_average(closes, int(name.split("_")[1]))
            if name.startswith("sma")
            else exponential_moving_average(closes, int(name.split("_")[1]))
        )
        series.append(
            {
                "name": INDICATOR_LABELS[name],
                "x": dates,
                # None leaves a gap for the days before the average has enough
                # history, rather than drawing a misleading flat start.
                "y": [None if v != v else round(float(v), 3) for v in values],
            }
        )

    forecast_note = ""
    if include_forecast:
        try:
            forecast = get_prediction(company)
        except ModelUnavailableError:
            forecast_note = (
                " The forecast was left off because the model for this company is "
                "unavailable or out of date; tell the user that."
            )
        except Exception:
            forecast_note = " The forecast could not be produced, so it was left off."
        else:
            # Starting the dashed line at the last recorded point keeps the two
            # series visually joined instead of leaving a gap between them.
            series.append(
                {
                    "name": "Forecast",
                    "x": [dates[-1]] + [d.strftime("%Y-%m-%d") for d in forecast["dates"]],
                    "y": [float(closes.iloc[-1])] + [round(v, 3) for v in forecast["values"]],
                    "dashed": True,
                    "color": series_color(1),
                }
            )
            forecast_note = " The dashed line is the model's forecast, not recorded data."

    period = f"{dates[0]} to {dates[-1]}"
    chart = chart_payload(
        title=f"{company} closing price",
        figure=line_figure(series, y_title="Close (TND)"),
        description=(
            f"Closing price of {company} from {period}"
            + (f", with {len(series) - 1} further series" if len(series) > 1 else "")
            + "."
        ),
    )

    return _drawn(
        tool_call_id,
        chart,
        f"Chart drawn: {company} closing price, {period}.{forecast_note}",
    )


@tool
def plot_comparison(
    companies: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Command:
    """Draw several Tunisian stocks on one chart to compare how they moved.

    Use this when the user wants to see two or more stocks side by side. Every
    line is rebased to 100 at the start of the period, so the chart compares how
    far each one moved rather than how expensive each one is: say so when you
    describe it, and quote real prices from the other tools if the user wants
    them.

    Args:
        companies: two to six ticker symbols.
        start_date: optional start of the period, as YYYY-MM-DD.
        end_date: optional end of the period, as YYYY-MM-DD.

    Returns:
        Confirmation that the chart was drawn, or the reason it was not.
    """
    if not companies or len(companies) < 2:
        return _failed(tool_call_id, tool_error("Name at least two companies to compare."))
    if len(companies) > MAX_SERIES:
        return _failed(
            tool_call_id,
            tool_error(
                f"Plot at most {MAX_SERIES} companies at once; {len(companies)} were given."
            ),
        )

    series = []
    failures = {}

    for company in companies:
        try:
            df = load_window(company, start_date, end_date)
        except Exception as e:
            failures[company] = handle_frame_error(company, e) or str(e)
            continue

        closes = df["cloture"]
        base = float(closes.iloc[0])
        if base == 0:
            failures[company] = "First close is zero, so it cannot be rebased."
            continue

        series.append(
            {
                "name": company,
                "x": [d.strftime("%Y-%m-%d") for d in df["date"]],
                "y": [round(float(c) / base * 100, 2) for c in closes],
            }
        )

    if len(series) < 2:
        return _failed(
            tool_call_id,
            tool_error(
                "Fewer than two companies could be loaded, so there is nothing to "
                "compare.",
                details=failures,
            ),
        )

    names = ", ".join(s["name"] for s in series)
    chart = chart_payload(
        title="Relative performance (start = 100)",
        figure=line_figure(series, y_title="Indexed to 100", hover_format=",.2f"),
        description=f"Relative performance of {names}, each rebased to 100 at the start.",
    )

    skipped = (
        f" {', '.join(failures)} could not be included."
        if failures
        else ""
    )
    return _drawn(
        tool_call_id,
        chart,
        f"Chart drawn: {names} rebased to 100 at the start of the period.{skipped}",
    )
