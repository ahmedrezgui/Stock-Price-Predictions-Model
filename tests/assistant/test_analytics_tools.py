import json
from unittest.mock import patch

import pandas as pd

from stock_assistant.tools import analyze_stock_performance, compare_stocks, top_movers
from market_data import CompanyNotFoundError


def price_history(closes, start="2024-01-01"):
    """Build a frame shaped like the processed CSVs, with the given closes."""
    dates = pd.date_range(start=start, periods=len(closes), freq="D")
    return pd.DataFrame({
        "date": dates,
        "ouverture": closes,
        "haut": closes,
        "bas": closes,
        "cloture": [float(c) for c in closes],
        "volume": [100] * len(closes),
    })


# ------------------------------------------------------------------- performance


@patch("stock_assistant.tools.frames.load_company_df")
def test_analyze_reports_return_and_risk(mock_load):
    mock_load.return_value = price_history([100, 120, 60, 130])

    payload = json.loads(analyze_stock_performance.invoke({"company": "AB"}))

    assert payload["company"] == "AB"
    assert payload["performance"]["total_return_pct"] == 30.0
    assert payload["performance"]["max_drawdown_pct"] == -50.0
    assert payload["period"]["trading_days"] == 4


@patch("stock_assistant.tools.frames.load_company_df")
def test_analyze_returns_null_for_metrics_the_period_cannot_support(mock_load):
    mock_load.return_value = price_history([100, 101, 102])

    payload = json.loads(analyze_stock_performance.invoke({"company": "AB"}))

    # Three days cannot produce a 20 or 50 day average, and saying so beats
    # inventing one.
    assert payload["indicators"]["sma_20"] is None
    assert payload["indicators"]["sma_50"] is None
    assert payload["indicators"]["rsi_14"] is None


@patch("stock_assistant.tools.frames.load_company_df")
def test_analyze_computes_indicators_when_there_is_enough_history(mock_load):
    mock_load.return_value = price_history(list(range(100, 160)))

    payload = json.loads(analyze_stock_performance.invoke({"company": "AB"}))

    assert payload["indicators"]["sma_20"] is not None
    assert payload["indicators"]["sma_50"] is not None
    assert payload["indicators"]["rsi_14"] is not None


@patch("stock_assistant.tools.frames.load_company_df")
def test_analyze_honours_the_date_range(mock_load):
    mock_load.return_value = price_history(list(range(100, 110)))

    payload = json.loads(
        analyze_stock_performance.invoke(
            {"company": "AB", "start_date": "2024-01-03", "end_date": "2024-01-05"}
        )
    )

    assert payload["period"]["trading_days"] == 3


@patch("stock_assistant.tools.frames.list_companies")
@patch("stock_assistant.tools.frames.load_company_df")
def test_analyze_reports_an_unknown_company(mock_load, mock_list):
    mock_load.side_effect = CompanyNotFoundError("nope")
    mock_list.return_value = ["AB", "BIAT"]

    payload = json.loads(analyze_stock_performance.invoke({"company": "NOPE"}))

    assert "NOPE" in payload["error"]
    assert payload["available_companies"] == "AB, BIAT"


@patch("stock_assistant.tools.frames.load_company_df")
def test_analyze_rejects_a_bad_date(mock_load):
    mock_load.return_value = price_history([100, 110])

    payload = json.loads(
        analyze_stock_performance.invoke({"company": "AB", "start_date": "yesterday"})
    )

    assert "YYYY-MM-DD" in payload["error"]


# ---------------------------------------------------------------------- compare


@patch("stock_assistant.tools.frames.load_company_df")
def test_compare_summarises_each_company(mock_load):
    mock_load.side_effect = lambda company: {
        "AB": price_history([100, 110, 120]),
        "CD": price_history([50, 45, 40]),
    }[company]

    payload = json.loads(compare_stocks.invoke({"companies": ["AB", "CD"]}))

    assert payload["companies"]["AB"]["total_return_pct"] == 20.0
    assert payload["companies"]["CD"]["total_return_pct"] == -20.0


@patch("stock_assistant.tools.frames.load_company_df")
def test_compare_correlates_daily_returns(mock_load):
    # CD is AB's mirror image, so their returns should be perfectly anticorrelated.
    mock_load.side_effect = lambda company: {
        "AB": price_history([100, 110, 100, 110]),
        "CD": price_history([100, 90, 100, 90]),
    }[company]

    payload = json.loads(compare_stocks.invoke({"companies": ["AB", "CD"]}))

    assert payload["correlation"]["AB"]["CD"] < -0.9
    assert payload["correlation"]["AB"]["AB"] == 1.0


@patch("stock_assistant.tools.frames.load_company_df")
def test_compare_carries_on_when_one_company_is_missing(mock_load):
    def load(company):
        if company == "NOPE":
            raise CompanyNotFoundError("nope")
        return price_history([100, 110])

    mock_load.side_effect = load

    payload = json.loads(compare_stocks.invoke({"companies": ["AB", "CD", "NOPE"]}))

    assert set(payload["companies"]) == {"AB", "CD"}
    assert "NOPE" in payload["unavailable"]


def test_compare_needs_companies():
    assert "at least two" in json.loads(compare_stocks.invoke({"companies": []}))["error"]


def test_compare_refuses_an_unreadable_number_of_companies():
    payload = json.loads(
        compare_stocks.invoke({"companies": [f"C{i}" for i in range(9)]})
    )

    assert "at most" in payload["error"]


@patch("stock_assistant.tools.frames.load_company_df")
def test_compare_reports_when_nothing_could_be_loaded(mock_load):
    mock_load.side_effect = CompanyNotFoundError("nope")

    payload = json.loads(compare_stocks.invoke({"companies": ["X", "Y"]}))

    assert "None of those companies" in payload["error"]


# ------------------------------------------------------------------- top movers


@patch("stock_assistant.tools.analytics_tools.list_companies")
@patch("stock_assistant.tools.frames.load_company_df")
def test_top_movers_ranks_gainers_and_losers(mock_load, mock_list):
    mock_list.return_value = ["UP", "DOWN", "FLAT"]
    mock_load.side_effect = lambda company: {
        "UP": price_history([100, 150]),
        "DOWN": price_history([100, 50]),
        "FLAT": price_history([100, 100]),
    }[company]

    payload = json.loads(top_movers.invoke({"period_days": 30, "limit": 1}))

    assert payload["gainers"][0]["company"] == "UP"
    assert payload["gainers"][0]["change_pct"] == 50.0
    assert payload["losers"][0]["company"] == "DOWN"


@patch("stock_assistant.tools.analytics_tools.list_companies")
@patch("stock_assistant.tools.frames.load_company_df")
def test_top_movers_skips_companies_without_enough_data(mock_load, mock_list):
    mock_list.return_value = ["GOOD", "THIN"]
    mock_load.side_effect = lambda company: {
        "GOOD": price_history([100, 150]),
        "THIN": price_history([100]),
    }[company]

    payload = json.loads(top_movers.invoke({}))

    assert payload["companies_ranked"] == 1
    assert payload["skipped"] == ["THIN"]


@patch("stock_assistant.tools.analytics_tools.list_companies")
@patch("stock_assistant.tools.frames.load_company_df")
def test_top_movers_measures_only_the_recent_window(mock_load, mock_list):
    mock_list.return_value = ["AB"]
    # Doubles over the first 10 days, then flat: a 5 day window should see ~0%.
    mock_load.return_value = price_history([100, 200] + [200] * 10)

    payload = json.loads(top_movers.invoke({"period_days": 5}))

    assert payload["gainers"][0]["change_pct"] == 0.0


def test_top_movers_rejects_a_meaningless_period():
    assert "at least 2" in json.loads(top_movers.invoke({"period_days": 1}))["error"]
