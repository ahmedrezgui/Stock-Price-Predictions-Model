import os
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

@patch.dict(os.environ, {"BACKEND_URL": "http://localhost:8000"})
@patch.dict(os.environ, {"FRONTEND_CACHE_EXPIRATION_TIME": "28800"})
@patch.dict(os.environ, {"WINDOW_SIZE": "1"})
@patch("requests_cache.CachedSession.get")

## TODO: We should Add other tests after implemeting the other widgets (1 test per widget at least)
def test_data_and_predict_widget(mock_get):

    def side_effect(url, *args, **kwargs):
        mock_response = MagicMock()
        if url.endswith("/companies"):
            mock_response.status_code = 200
            mock_response.json.return_value = ["AB"]
        elif "/stock/AB" in url:
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "columns": ["date", "cloture"],
                "data": [
                    {"date": "2020-01-01", "cloture": 100},
                    {"date": "2020-01-02", "cloture": 102},
                ]
            }
        return mock_response

    mock_get.side_effect = side_effect

    at = AppTest.from_file("apps/front/home.py")
    at.run(timeout=10)

    at.button._list[1].click()
    at.run(timeout=10)

    at.selectbox[0].select("AB")
    at.session_state["Select a company"] = "AB"
    at.run(timeout=10)

    next(s for s in at.selectbox if s.label == "Bar interval").select("Month").run(timeout=10)
    assert not at.exception
    next(s for s in at.selectbox if s.label == "Bar interval").select("Year").run(timeout=10)
    assert not at.exception
    # Assert no errors — basic health check
    assert len(at.error) == 0


@patch.dict(os.environ, {"BACKEND_URL": "http://localhost:8002", "FRONTEND_CACHE_EXPIRATION_TIME": "300", "WINDOW_SIZE": "20"})
@patch("requests_cache.CachedSession.get")
def test_stock_service_failure_shows_message_instead_of_traceback(mock_get):
    from requests.exceptions import HTTPError

    companies = MagicMock()
    companies.json.return_value = ["AB"]
    stock = MagicMock()
    stock.raise_for_status.side_effect = HTTPError("503 Service Unavailable")
    mock_get.side_effect = lambda url, **kwargs: companies if url.endswith("/companies") else stock
    at = AppTest.from_file("apps/front/home.py")
    at.query_params["page"] = "Data & Predict"
    at.run(timeout=10)
    assert not at.exception
    assert len(at.error) == 1
    assert "AB" in at.error[0].value


@patch("requests.post")
def test_chat_suggestion_renders_answer_and_new_session(mock_post):
    import json
    response = mock_post.return_value.__enter__.return_value
    response.iter_lines.return_value = [json.dumps(e).encode() for e in [
        {"type": "progress", "id": "tool-1", "label": "Fetching the deployed forecast", "status": "running"},
        {"type": "progress", "id": "tool-1", "label": "Fetching the deployed forecast", "status": "completed"},
        {"type": "result", "text": "Forecast evidence", "charts": [], "sources": []},
    ]]
    at = AppTest.from_file("apps/front/home.py").run(timeout=10)
    at.switch_page("pages/chatbot.py").run(timeout=10)
    assert not at.exception
    next(b for b in at.button if b.label == "Explore a forecast").click().run(timeout=10)
    assert not at.exception
    assert mock_post.call_args.kwargs["json"]["question"].startswith("Plot BIAT")
    assert any(m.value == "Forecast evidence" for m in at.markdown)
    thread = at.session_state["thread_id"]
    next(b for b in at.button if b.label == "New Chat").click().run(timeout=10)
    assert not at.exception
    assert at.session_state["thread_id"] != thread
    assert len(at.session_state["chats"]) == 1
    next(b for b in at.button if b.key == f"chat_{thread}").click().run(timeout=10)
    assert not at.exception
    assert at.session_state["thread_id"] == thread
    assert any(m.value == "Forecast evidence" for m in at.markdown)
    next(b for b in at.button if b.key == f"delete_{thread}").click().run(timeout=10)
    assert not at.exception
    assert not at.session_state["chats"]
