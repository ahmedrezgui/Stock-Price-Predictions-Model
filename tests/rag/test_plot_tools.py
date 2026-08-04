import json
from unittest.mock import patch

import pandas as pd
import pytest

from src.rag.tools import plot_comparison, plot_stock
from src.rag.tools.charts import MAX_SERIES, SERIES_COLORS, line_figure
from src.services import CompanyNotFoundError, ModelUnavailableError


def price_history(closes, start="2024-01-01"):
    dates = pd.date_range(start=start, periods=len(closes), freq="D")
    return pd.DataFrame({
        "date": dates,
        "ouverture": closes,
        "haut": closes,
        "bas": closes,
        "cloture": [float(c) for c in closes],
        "volume": [100] * len(closes),
    })


def call(tool, args):
    """Invoke a tool the way the graph does, so the tool call id is injected."""
    return tool.invoke(
        {"name": tool.name, "args": args, "id": "call_1", "type": "tool_call"}
    )


def tool_message(command):
    return command.update["messages"][0].content


def charts(command):
    return command.update.get("charts", [])


# ------------------------------------------------------------------- the figure


def test_line_figure_colours_series_in_slot_order():
    figure = line_figure(
        [
            {"name": "a", "x": ["2024-01-01"], "y": [1]},
            {"name": "b", "x": ["2024-01-01"], "y": [2]},
        ],
        y_title="Close",
    )

    assert figure["data"][0]["line"]["color"] == SERIES_COLORS[0]
    assert figure["data"][1]["line"]["color"] == SERIES_COLORS[1]


def test_line_figure_shows_a_legend_only_when_series_share_the_chart():
    single = line_figure([{"name": "a", "x": ["2024-01-01"], "y": [1]}], y_title="c")
    paired = line_figure(
        [
            {"name": "a", "x": ["2024-01-01"], "y": [1]},
            {"name": "b", "x": ["2024-01-01"], "y": [2]},
        ],
        y_title="c",
    )

    assert single["layout"]["showlegend"] is False
    assert paired["layout"]["showlegend"] is True


def test_line_figure_marks_projected_series_with_a_dash():
    figure = line_figure(
        [{"name": "forecast", "x": ["2024-01-01"], "y": [1], "dashed": True}],
        y_title="c",
    )

    assert figure["data"][0]["line"]["dash"] == "dash"


def test_line_figure_is_transparent_so_either_theme_shows_through():
    figure = line_figure([{"name": "a", "x": ["2024-01-01"], "y": [1]}], y_title="c")

    assert figure["layout"]["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert figure["layout"]["plot_bgcolor"] == "rgba(0,0,0,0)"


def test_line_figure_is_plain_json():
    figure = line_figure([{"name": "a", "x": ["2024-01-01"], "y": [1.5]}], y_title="c")

    # It travels to the browser as JSON, so it must survive a round trip.
    assert json.loads(json.dumps(figure)) == figure


# -------------------------------------------------------------------- one stock


@patch("src.rag.tools.frames.load_company_df")
def test_plot_stock_puts_a_chart_in_state_and_a_sentence_in_the_conversation(mock_load):
    mock_load.return_value = price_history([100, 110, 120])

    command = call(plot_stock, {"company": "AB"})

    assert len(charts(command)) == 1
    assert charts(command)[0]["title"] == "AB closing price"
    # The model is told the chart exists; the figure itself never reaches it.
    assert "Chart drawn" in tool_message(command)
    assert "figure" not in tool_message(command)


@patch("src.rag.tools.frames.load_company_df")
def test_plot_stock_draws_requested_indicators(mock_load):
    mock_load.return_value = price_history(list(range(100, 160)))

    command = call(plot_stock, {"company": "AB", "indicators": ["sma_20", "ema_20"]})

    names = [trace["name"] for trace in charts(command)[0]["figure"]["data"]]
    assert names == ["AB close", "20-day average", "20-day EMA"]


@patch("src.rag.tools.frames.load_company_df")
def test_plot_stock_rejects_an_unknown_indicator(mock_load):
    mock_load.return_value = price_history([100, 110])

    command = call(plot_stock, {"company": "AB", "indicators": ["macd"]})

    assert "Unsupported indicators" in tool_message(command)
    assert charts(command) == []


@patch("src.rag.tools.prediction_tools.get_prediction")
@patch("src.rag.tools.plot_tools.get_prediction")
@patch("src.rag.tools.frames.load_company_df")
def test_plot_stock_appends_the_forecast_as_a_dashed_line(
    mock_load, mock_forecast, _unused
):
    mock_load.return_value = price_history([100, 110, 120])
    mock_forecast.return_value = {
        "company": "AB",
        "last_close": 120.0,
        "last_date": pd.Timestamp("2024-01-03"),
        "dates": [pd.Timestamp("2024-01-04")],
        "values": [125.0],
    }

    command = call(plot_stock, {"company": "AB", "include_forecast": True})

    forecast_trace = charts(command)[0]["figure"]["data"][-1]
    assert forecast_trace["name"] == "Forecast"
    assert forecast_trace["line"]["dash"] == "dash"
    # It starts at the last recorded point, so the two lines join up.
    assert forecast_trace["y"][0] == 120.0
    assert "dashed line is the model's forecast" in tool_message(command)


@patch("src.rag.tools.plot_tools.get_prediction")
@patch("src.rag.tools.frames.load_company_df")
def test_plot_stock_still_draws_when_the_forecast_is_unavailable(
    mock_load, mock_forecast
):
    mock_load.return_value = price_history([100, 110, 120])
    mock_forecast.side_effect = ModelUnavailableError("stale")

    command = call(plot_stock, {"company": "AB", "include_forecast": True})

    assert len(charts(command)) == 1
    assert len(charts(command)[0]["figure"]["data"]) == 1
    assert "unavailable or out of date" in tool_message(command)


@patch("src.rag.tools.frames.list_companies")
@patch("src.rag.tools.frames.load_company_df")
def test_plot_stock_reports_an_unknown_company(mock_load, mock_list):
    mock_load.side_effect = CompanyNotFoundError("nope")
    mock_list.return_value = ["AB"]

    command = call(plot_stock, {"company": "NOPE"})

    assert charts(command) == []
    assert "NOPE" in tool_message(command)


# ------------------------------------------------------------------ comparison


@patch("src.rag.tools.frames.load_company_df")
def test_plot_comparison_rebases_every_line_to_100(mock_load):
    mock_load.side_effect = lambda company: {
        "AB": price_history([100, 150]),
        "CD": price_history([10, 5]),
    }[company]

    command = call(plot_comparison, {"companies": ["AB", "CD"]})

    traces = {
        trace["name"]: list(trace["y"])
        for trace in charts(command)[0]["figure"]["data"]
    }
    # Rebasing is what lets prices of different magnitudes share one axis.
    assert traces["AB"] == [100.0, 150.0]
    assert traces["CD"] == [100.0, 50.0]


@patch("src.rag.tools.frames.load_company_df")
def test_plot_comparison_needs_two_companies(mock_load):
    command = call(plot_comparison, {"companies": ["AB"]})

    assert "at least two" in tool_message(command)
    assert charts(command) == []


@patch("src.rag.tools.frames.load_company_df")
def test_plot_comparison_refuses_more_series_than_the_palette_holds(mock_load):
    command = call(
        plot_comparison, {"companies": [f"C{i}" for i in range(MAX_SERIES + 1)]}
    )

    assert "at most" in tool_message(command)


@patch("src.rag.tools.frames.load_company_df")
def test_plot_comparison_notes_a_company_it_could_not_draw(mock_load):
    def load(company):
        if company == "NOPE":
            raise CompanyNotFoundError("nope")
        return price_history([100, 110])

    mock_load.side_effect = load

    command = call(plot_comparison, {"companies": ["AB", "CD", "NOPE"]})

    assert len(charts(command)[0]["figure"]["data"]) == 2
    assert "NOPE could not be included" in tool_message(command)


@patch("src.rag.tools.frames.load_company_df")
def test_plot_comparison_gives_up_when_too_few_survive(mock_load):
    def load(company):
        if company == "AB":
            return price_history([100, 110])
        raise CompanyNotFoundError("nope")

    mock_load.side_effect = load

    command = call(plot_comparison, {"companies": ["AB", "NOPE"]})

    assert charts(command) == []
    assert "Fewer than two" in tool_message(command)
