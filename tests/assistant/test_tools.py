import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.documents import Document

from stock_assistant.tools import (
    get_stock_data,
    list_available_companies,
    predict_stock,
    retrieve_news,
)
from market_data import (
    CompanyNotFoundError,
    DataUnavailableError,
    ModelUnavailableError,
)


@pytest.fixture(autouse=True)
def keep_every_passage():
    """
    Relevance grading is tested in test_document_grader.py.

    Everywhere else it should pass passages through untouched, so that a test
    about retrieval is not also a test about grading.
    """
    with patch(
        "stock_assistant.tools.retrieval_tools.keep_relevant",
        side_effect=lambda query, documents: documents,
    ):
        yield


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


@patch("stock_assistant.tools.retrieval_tools.get_vector_store")
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


@patch("stock_assistant.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_filters_to_news_source(mock_store):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_store.return_value.as_retriever.return_value = retriever

    retrieve_news.invoke({"query": "market", "k": 3})

    kwargs = mock_store.return_value.as_retriever.call_args.kwargs
    assert kwargs["search_kwargs"] == {"k": 3, "filter": {"source": "news"}}


@patch("stock_assistant.tools.retrieval_tools.get_vector_store")
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


@patch("stock_assistant.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_accepts_an_open_ended_range(mock_store):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_store.return_value.as_retriever.return_value = retriever

    retrieve_news.invoke({"query": "market", "date_from": "2024-03-01"})

    metadata_filter = mock_store.return_value.as_retriever.call_args.kwargs[
        "search_kwargs"
    ]["filter"]
    assert metadata_filter["timestamp"] == {"$gte": 1709251200}


@patch("stock_assistant.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_rejects_a_misformatted_date(mock_store):
    payload = json.loads(
        retrieve_news.invoke({"query": "market", "date_from": "01/03/2024"})
    )

    assert "YYYY-MM-DD" in payload["error"]
    mock_store.assert_not_called()


@patch("stock_assistant.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_reports_failure_without_raising(mock_store):
    mock_store.side_effect = RuntimeError("pinecone down")

    payload = json.loads(retrieve_news.invoke({"query": "market"}))

    assert payload["documents"] == []
    assert "pinecone down" in payload["error"]


@patch("stock_assistant.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_drops_passages_the_grader_rejects(mock_store):
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(page_content="relevant", metadata={"title": "A", "link": "https://x/a"}),
        Document(page_content="unrelated", metadata={"title": "B", "link": "https://x/b"}),
    ]
    mock_store.return_value.as_retriever.return_value = retriever

    with patch(
        "stock_assistant.tools.retrieval_tools.keep_relevant",
        side_effect=lambda query, documents: documents[:1],
    ):
        payload = json.loads(retrieve_news.invoke({"query": "market"}))

    assert [d["title"] for d in payload["documents"]] == ["A"]
    assert "1 of 2 retrieved passages were dropped" in payload["note"]


@patch("stock_assistant.tools.retrieval_tools.get_vector_store")
def test_retrieve_news_says_so_when_grading_leaves_nothing(mock_store):
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(page_content="unrelated", metadata={"title": "B", "link": "https://x/b"})
    ]
    mock_store.return_value.as_retriever.return_value = retriever

    with patch(
        "stock_assistant.tools.retrieval_tools.keep_relevant",
        side_effect=lambda query, documents: [],
    ):
        payload = json.loads(retrieve_news.invoke({"query": "market"}))

    assert payload["documents"] == []
    # The agent is told to change tack rather than left to guess.
    assert "search the web instead" in payload["note"]


# --------------------------------------------------------------------- stock data


@patch("stock_assistant.tools.stock_data_tools.list_companies")
def test_list_available_companies_is_sorted(mock_list):
    mock_list.return_value = ["BIAT", "AB"]

    assert json.loads(list_available_companies.invoke({}))["companies"] == ["AB", "BIAT"]


@patch("stock_assistant.tools.stock_data_tools.load_company_df")
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


@patch("stock_assistant.tools.stock_data_tools.load_company_df")
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


@patch("stock_assistant.tools.stock_data_tools.load_company_df")
def test_get_stock_data_caps_the_rows_it_returns(mock_load):
    mock_load.return_value = _price_history(days=40)

    payload = json.loads(get_stock_data.invoke({"company": "AB"}))

    assert len(payload["rows"]) == 30
    assert payload["period"]["trading_days"] == 40
    assert "rows_note" in payload
    # The summary still covers everything, not just the rows shown.
    assert payload["summary"]["first_close"] == 100


@patch("stock_assistant.tools.stock_data_tools.list_companies")
@patch("stock_assistant.tools.stock_data_tools.load_company_df")
def test_get_stock_data_suggests_symbols_it_knows(mock_load, mock_list):
    mock_load.side_effect = CompanyNotFoundError("Company not found")
    mock_list.return_value = ["AB", "BIAT"]

    payload = json.loads(get_stock_data.invoke({"company": "NOPE"}))

    assert "NOPE" in payload["error"]
    assert payload["available_companies"] == "AB, BIAT"


@patch("stock_assistant.tools.stock_data_tools.load_company_df")
def test_get_stock_data_rejects_a_bad_date(mock_load):
    mock_load.return_value = _price_history()

    payload = json.loads(
        get_stock_data.invoke({"company": "AB", "start_date": "last tuesday"})
    )

    assert "YYYY-MM-DD" in payload["error"]


@patch("stock_assistant.tools.stock_data_tools.load_company_df")
def test_get_stock_data_reports_an_empty_range(mock_load):
    mock_load.return_value = _price_history(days=5)

    payload = json.loads(
        get_stock_data.invoke({"company": "AB", "start_date": "2030-01-01"})
    )

    assert "No rows recorded" in payload["error"]


@patch("stock_assistant.tools.stock_data_tools.load_company_df")
def test_get_stock_data_reports_a_failed_pull(mock_load):
    mock_load.side_effect = DataUnavailableError("dvc pull failed")

    payload = json.loads(get_stock_data.invoke({"company": "AB"}))

    assert payload["error"] == "dvc pull failed"


# --------------------------------------------------------------------- prediction


@patch("stock_assistant.tools.prediction_tools.get_prediction")
def test_predict_stock_preserves_service_evidence(mock_prediction):
    mock_prediction.return_value = {
        "symbol": "BIAT", "as_of": "2026-10-01", "forecast_for": None,
        "model_version": "v1", "deployment_status": "unapproved",
        "forecast_horizon_observations": 5, "horizon_basis": "observations",
        "predicted_log_return": 0.01234567, "implied_price": 101.2,
    }
    payload = json.loads(predict_stock.invoke({"company": "BIAT", "horizon": 5}))
    assert payload["predicted_log_return"] == 0.01234567
    assert payload["forecast_for"] is None
    assert payload["deployment_status"] == "unapproved"
    assert "not investment advice" in payload["disclaimer"]
    mock_prediction.assert_called_once_with("BIAT", 5, None)


@patch("stock_assistant.tools.prediction_tools.get_prediction")
def test_predict_stock_explains_service_failure(mock_prediction):
    from stock_assistant.forecast_client import ForecastServiceError
    mock_prediction.side_effect = ForecastServiceError("Forecast service returned HTTP 503.")
    payload = json.loads(predict_stock.invoke({"company": "BIAT"}))
    assert "503" in payload["error"]
    assert "get_stock_data" in payload["note"]
