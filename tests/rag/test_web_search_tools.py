import json
import os
from unittest.mock import MagicMock, patch

from src.rag.tools.web_search_tools import is_configured, run_search, web_search

RESPONSE = {
    "results": [
        {
            "title": "Tunis market rises",
            "url": "https://example.com/a",
            "content": "The index gained 1%.",
            "published_date": "2024-03-01",
        }
    ]
}


@patch.dict(os.environ, {"TAVILY_API_KEY": "key"})
@patch("src.rag.tools.web_search_tools.get_client")
def test_search_returns_documents_shaped_like_the_archive(mock_client):
    mock_client.return_value.search.return_value = RESPONSE

    payload = json.loads(web_search.invoke({"query": "market"}))

    # The same shape as retrieve_news, so citations are collected the same way.
    assert payload["documents"] == [{
        "title": "Tunis market rises",
        "date": "2024-03-01",
        "link": "https://example.com/a",
        "content": "The index gained 1%.",
    }]


@patch.dict(os.environ, {"TAVILY_API_KEY": "key"})
@patch("src.rag.tools.web_search_tools.get_client")
def test_search_adds_tunisian_context_to_the_query(mock_client):
    mock_client.return_value.search.return_value = {"results": []}

    web_search.invoke({"query": "inflation"})

    assert "Tunisian" in mock_client.return_value.search.call_args.kwargs["query"]


@patch.dict(os.environ, {}, clear=True)
def test_search_reports_that_it_is_unconfigured_rather_than_failing():
    payload = json.loads(web_search.invoke({"query": "market"}))

    assert payload["documents"] == []
    assert "not configured" in payload["error"]


@patch.dict(os.environ, {"TAVILY_API_KEY": "key"})
@patch("src.rag.tools.web_search_tools.get_client")
def test_search_reports_a_provider_failure_without_raising(mock_client):
    mock_client.return_value.search.side_effect = RuntimeError("tavily down")

    payload = json.loads(web_search.invoke({"query": "market"}))

    assert payload["documents"] == []
    assert "tavily down" in payload["error"]


@patch.dict(os.environ, {"TAVILY_API_KEY": "key"})
def test_is_configured_follows_the_key():
    assert is_configured() is True


@patch.dict(os.environ, {}, clear=True)
def test_is_configured_is_false_without_a_key():
    assert is_configured() is False


@patch.dict(os.environ, {"TAVILY_API_KEY": "key"})
@patch("src.rag.tools.web_search_tools.get_client")
def test_run_search_honours_the_result_count(mock_client):
    mock_client.return_value.search.return_value = {"results": []}

    run_search("market", max_results=2)

    assert mock_client.return_value.search.call_args.kwargs["max_results"] == 2
