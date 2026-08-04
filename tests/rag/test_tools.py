import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.documents import Document

from src.rag.tools import (
    get_stock_data,
    list_available_companies,
    predict_stock,
    retrieve_news,
)
from src.services import (
    CompanyNotFoundError,
    DataUnavailableError,
    ModelUnavailableError,
)


def _price_history(days=5, start="2024-01-01"):
    dates = pd.date_range(start=start, periods=days, freq="D")
    return pd.DataFrame({
        "date": dates,
        "ouverture": range(100, 100 + days),
        "haut": range(101, 101 + days),
        "bas": range(99, 99 + days),
        "cloture": range(100, 100 + days),
        "volume": [10] * days,
    })


# --------------------------------------------------------------------------- news


@patch("src.rag.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_returns_documents_with_metadata(mock_store):
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(
            page_content="The market rose.",
            metadata={
                "title": "Market up",
                "date": "01/03/2024",
                "link": "https://example.com/a",
                "source": "news",
            },
        )
    ]
    mock_store.return_value.as_retriever.return_value = retriever

    payload = json.loads(retrieve_news.invoke({"query": "market"}))

    assert payload["documents"] == [{
        "title": "Market up",
        "date": "01/03/2024",
        "link": "https://example.com/a",
        "content": "The market rose.",
    }]


@patch("src.rag.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_filters_to_news_source(mock_store):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_store.return_value.as_retriever.return_value = retriever

    retrieve_news.invoke({"query": "market", "k": 3})

    kwargs = mock_store.return_value.as_retriever.call_args.kwargs
    assert kwargs["search_kwargs"] == {"k": 3, "filter": {"source": "news"}}


@patch("src.rag.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_filters_by_date_range(mock_store):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_store.return_value.as_retriever.return_value = retriever

    retrieve_news.invoke(
        {"query": "market", "date_from": "2024-03-01", "date_to": "2024-03-15"}
    )

    metadata_filter = mock_store.return_value.as_retriever.call_args.kwargs[
        "search_kwargs"
    ]["filter"]
    assert metadata_filter["source"] == "news"
    # 2024-03-01T00:00:00Z through the last second of 2024-03-15, so the closing
    # day is included rather than cut off at midnight.
    assert metadata_filter["timestamp"]["$gte"] == 1709251200
    assert metadata_filter["timestamp"]["$lte"] == 1710547199


@patch("src.rag.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_accepts_an_open_ended_range(mock_store):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_store.return_value.as_retriever.return_value = retriever

    retrieve_news.invoke({"query": "market", "date_from": "2024-03-01"})

    metadata_filter = mock_store.return_value.as_retriever.call_args.kwargs[
        "search_kwargs"
    ]["filter"]
    assert metadata_filter["timestamp"] == {"$gte": 1709251200}


@patch("src.rag.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_rejects_a_misformatted_date(mock_store):
    payload = json.loads(
        retrieve_news.invoke({"query": "market", "date_from": "01/03/2024"})
    )

    assert "YYYY-MM-DD" in payload["error"]
    mock_store.assert_not_called()


@patch("src.rag.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_reports_failure_without_raising(mock_store):
    mock_store.side_effect = RuntimeError("pinecone down")

    payload = json.loads(retrieve_news.invoke({"query": "market"}))

    assert payload["documents"] == []
    assert "pinecone down" in payload["error"]


# --------------------------------------------------------------------- stock data


@patch("src.rag.tools.stock_data_tools.list_companies")
def test_list_available_companies_is_sorted(mock_list):
    mock_list.return_value = ["BIAT", "AB"]

    assert json.loads(list_available_companies.invoke({}))["companies"] == ["AB", "BIAT"]


@patch("src.rag.tools.stock_data_tools.load_company_df")
def test_get_stock_data_summarises_the_period(mock_load):
    mock_load.return_value = _price_history(days=5)

    payload = json.loads(get_stock_data.invoke({"company": "AB"}))

    assert payload["period"] == {
        "from": "2024-01-01",
        "to": "2024-01-05",
        "trading_days": 5,
    }
    assert payload["summary"]["first_close"] == 100
    assert payload["summary"]["last_close"] == 104
    assert payload["summary"]["change_pct"] == 4.0
    assert payload["rows"][0] == {
        "date": "2024-01-01",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 10.0,
    }


@patch("src.rag.tools.stock_data_tools.load_company_df")
def test_get_stock_data_honours_the_date_range(mock_load):
    mock_load.return_value = _price_history(days=10)

    payload = json.loads(
        get_stock_data.invoke(
            {"company": "AB", "start_date": "2024-01-03", "end_date": "2024-01-05"}
        )
    )

    assert payload["period"]["trading_days"] == 3
    assert payload["period"]["from"] == "2024-01-03"
    assert payload["period"]["to"] == "2024-01-05"


@patch("src.rag.tools.stock_data_tools.load_company_df")
def test_get_stock_data_caps_the_rows_it_returns(mock_load):
    mock_load.return_value = _price_history(days=40)

    payload = json.loads(get_stock_data.invoke({"company": "AB"}))

    assert len(payload["rows"]) == 30
    assert payload["period"]["trading_days"] == 40
    assert "rows_note" in payload
    # The summary still covers everything, not just the rows shown.
    assert payload["summary"]["first_close"] == 100


@patch("src.rag.tools.stock_data_tools.list_companies")
@patch("src.rag.tools.stock_data_tools.load_company_df")
def test_get_stock_data_suggests_symbols_it_knows(mock_load, mock_list):
    mock_load.side_effect = CompanyNotFoundError("Company not found")
    mock_list.return_value = ["AB", "BIAT"]

    payload = json.loads(get_stock_data.invoke({"company": "NOPE"}))

    assert "NOPE" in payload["error"]
    assert payload["available_companies"] == "AB, BIAT"


@patch("src.rag.tools.stock_data_tools.load_company_df")
def test_get_stock_data_rejects_a_bad_date(mock_load):
    mock_load.return_value = _price_history()

    payload = json.loads(
        get_stock_data.invoke({"company": "AB", "start_date": "last tuesday"})
    )

    assert "YYYY-MM-DD" in payload["error"]


@patch("src.rag.tools.stock_data_tools.load_company_df")
def test_get_stock_data_reports_an_empty_range(mock_load):
    mock_load.return_value = _price_history(days=5)

    payload = json.loads(
        get_stock_data.invoke({"company": "AB", "start_date": "2030-01-01"})
    )

    assert "No rows recorded" in payload["error"]


@patch("src.rag.tools.stock_data_tools.load_company_df")
def test_get_stock_data_reports_a_failed_pull(mock_load):
    mock_load.side_effect = DataUnavailableError("dvc pull failed")

    payload = json.loads(get_stock_data.invoke({"company": "AB"}))

    assert payload["error"] == "dvc pull failed"


# --------------------------------------------------------------------- prediction


@patch("src.rag.tools.prediction_tools.get_prediction")
def test_predict_stock_returns_dated_values_and_disclaimer(mock_prediction):
    mock_prediction.return_value = {
        "company": "AB",
        "last_close": 100.0,
        "last_date": pd.Timestamp("2024-01-05"),
        "dates": [pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-07")],
        "values": [101.234567, 102.0],
    }

    payload = json.loads(predict_stock.invoke({"company": "AB", "horizon_days": 2}))

    assert payload["last_recorded_date"] == "2024-01-05"
    assert payload["forecast"] == [
        {"date": "2024-01-06", "predicted_close": 101.235},
        {"date": "2024-01-07", "predicted_close": 102.0},
    ]
    assert "not investment advice" in payload["disclaimer"]


@patch("src.rag.tools.prediction_tools.get_prediction")
def test_predict_stock_explains_an_unusable_model(mock_prediction):
    mock_prediction.side_effect = ModelUnavailableError("Model unavailable, Try later")

    payload = json.loads(predict_stock.invoke({"company": "AB"}))

    # The agent should learn it can still fall back to historical data.
    assert "get_stock_data" in payload["error"]


@patch("src.rag.tools.prediction_tools.get_prediction")
def test_predict_stock_reports_an_unknown_company(mock_prediction):
    mock_prediction.side_effect = CompanyNotFoundError("Company not found")

    payload = json.loads(predict_stock.invoke({"company": "NOPE"}))

    assert "NOPE" in payload["error"]
