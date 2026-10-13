"""Service boundaries, provider selection and durable GenAI tracing."""

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from stock_assistant.api import create_app
from stock_assistant.forecast_client import ForecastServiceError, get_prediction
from stock_assistant.providers import create_chat_model
from stock_assistant.graph import collect_sources


FORECAST = {
    "symbol": "BIAT",
    "model": "lstm",
    "model_version": "v1",
    "as_of": "2026-10-01",
    "forecast_for": None,
    "forecast_horizon_observations": 1,
    "horizon_basis": "observations",
    "predicted_log_return": 0.02,
    "implied_price": 102,
    "deployment_status": "unapproved",
    "approval_id": None,
}


def test_endpoint_authentication_and_evidence(monkeypatch):
    monkeypatch.setenv("FORECAST_API_KEY", "test-secret")
    monkeypatch.setenv("ASSISTANT_FORECAST_URL", "http://forecast:8000")

    def handle(request):
        assert str(request.url) == "http://forecast:8000/v1/predict"
        assert request.headers["X-API-Key"] == "test-secret"
        assert json.loads(request.content) == {"symbol": "BIAT", "model": "lstm", "horizon": 1, "task": "return"}
        return httpx.Response(200, json=FORECAST)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    with patch("stock_assistant.forecast_client.httpx.Client", return_value=client):
        assert get_prediction("biat") == FORECAST


@pytest.mark.parametrize(
    "status,payload", [(401, {"secret": "sensitive"}), (503, {}), (200, {}), (200, {**FORECAST, "symbol": "OTHER"})]
)
def test_endpoint_fails_closed_without_leaking_response(status, payload, monkeypatch):
    monkeypatch.setenv("ASSISTANT_FORECAST_URL", "https://forecast.example")
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(status, json=payload)))
    with patch("stock_assistant.forecast_client.httpx.Client", return_value=client):
        with pytest.raises(ForecastServiceError) as error:
            get_prediction("BIAT")
    assert "sensitive" not in str(error.value)


def test_invalid_horizon_never_calls_http():
    with patch("stock_assistant.forecast_client.httpx.Client") as client:
        with pytest.raises(ValueError):
            get_prediction("BIAT", 2)
        client.assert_not_called()


def test_catalog_uses_configured_service(monkeypatch):
    from stock_assistant.forecast_client import get_models

    monkeypatch.setenv("ASSISTANT_FORECAST_URL", "http://forecast:8000")

    def handle(request):
        assert str(request.url) == "http://forecast:8000/models"
        return httpx.Response(200, json=[{"model_version": "v1"}])

    with patch(
        "stock_assistant.forecast_client.httpx.Client", return_value=httpx.Client(transport=httpx.MockTransport(handle))
    ):
        assert get_models() == [{"model_version": "v1"}]


def test_history_reads_local_decimal_comma_bars_and_blocks_path_traversal(tmp_path, monkeypatch):
    from market_data.history import list_companies, load_company_df
    from market_data import CompanyNotFoundError

    monkeypatch.setenv("ASSISTANT_DATA_ROOT", str(tmp_path))
    (tmp_path / "BIAT.csv").write_text(
        "symbole;date;ouverture;haut;bas;cloture;volume\nBIAT;02/01/2024;10,0;11,0;9,0;10,5;100\n"
    )
    assert list_companies() == ["BIAT"]
    frame = load_company_df("BIAT")
    assert frame.iloc[0]["cloture"] == 10.5
    assert str(frame.iloc[0]["date"].date()) == "2024-01-02"
    with pytest.raises(CompanyNotFoundError):
        load_company_df("../secret")


def test_simple_model_defaults_to_main_provider_model():
    fake = MagicMock()
    with patch.dict(os.environ, {"ASSISTANT_PROVIDER": "ollama", "ASSISTANT_MODEL": "local-model"}, clear=True):
        with patch.dict("sys.modules", {"langchain_ollama": fake}):
            create_chat_model(simple=True)
    assert fake.ChatOllama.call_args.kwargs["model"] == "local-model"


@pytest.mark.parametrize(
    "provider,module,constructor",
    [
        ("gemini", "langchain_google_genai", "ChatGoogleGenerativeAI"),
        ("ollama", "langchain_ollama", "ChatOllama"),
        ("azure", "langchain_openai", "AzureChatOpenAI"),
    ],
)
def test_provider_factory_is_lazy_and_explicit(provider, module, constructor):
    fake_module = MagicMock()
    env = {
        "ASSISTANT_PROVIDER": provider,
        "ASSISTANT_MODEL": "tool-model",
        "GOOGLE_API_KEY": "test",
        "AZURE_OPENAI_ENDPOINT": "https://example.test",
        "AZURE_OPENAI_API_KEY": "test",
        "AZURE_OPENAI_API_VERSION": "configured-version",
    }
    with patch.dict(os.environ, env, clear=True), patch.dict("sys.modules", {module: fake_module}):
        assert create_chat_model() is getattr(fake_module, constructor).return_value
        getattr(fake_module, constructor).assert_called_once()


def test_current_turn_does_not_inherit_old_citations():
    tool = ToolMessage(
        content=json.dumps({"documents": [{"link": "https://old.test"}]}), name="web_search", tool_call_id="old"
    )
    assert (
        collect_sources([HumanMessage("Old question"), tool, AIMessage("Answer"), HumanMessage("New question")]) == []
    )


def test_api_returns_charts_and_preserves_thread(monkeypatch):
    route = "/chat"
    monkeypatch.setenv("ASSISTANT_TRACING_ENABLED", "false")
    graph = MagicMock()
    graph.invoke.return_value = {"generation": "Answer", "charts": [{"title": "BIAT"}], "sources": []}
    with TestClient(create_app(graph)) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert "post" in paths[route]
        assert "/rag" not in paths
        assert client.post("/rag", json={"question": "BIAT?"}).status_code == 404
        response = client.post(route, json={"question": "Plot BIAT", "thread_id": "thread-1"})
        assert response.status_code == 200
        assert response.json()["charts"] == [{"title": "BIAT"}]
        assert response.json()["thread_id"] == "thread-1"
        assert client.post(route, json={"question": " "}).status_code == 422
        graph.invoke.side_effect = RuntimeError("secret provider detail")
        response = client.post(route, json={"question": "BIAT?"})
        assert response.status_code == 503
        assert "secret" not in response.text


def test_mlflow_persists_langgraph_trace_in_genai_experiment(tmp_path, monkeypatch):
    import mlflow
    import mlflow.langchain
    from stock_assistant import observability

    uri = "sqlite:///" + (tmp_path / "traces.db").as_posix()
    previous = mlflow.get_tracking_uri()
    # MLflow setters mutate environment variables; isolate them from later tests.
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME"):
        monkeypatch.setenv(key, os.getenv(key, ""))
    monkeypatch.setenv("ASSISTANT_TRACING_ENABLED", "true")
    monkeypatch.setenv("ASSISTANT_MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("ASSISTANT_MLFLOW_EXPERIMENT", "assistant-test")
    monkeypatch.setattr(observability, "_configured", None)
    try:
        observability.configure_tracking()
        workflow = StateGraph(MessagesState)
        workflow.add_node("answer", lambda state: {"messages": [AIMessage("Offline trace check")]})
        workflow.add_edge(START, "answer")
        workflow.add_edge("answer", END)
        workflow.compile().invoke({"messages": [HumanMessage("test")]})
        mlflow.flush_trace_async_logging()
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name("assistant-test")
        assert experiment.tags["mlflow.experimentKind"] == "genai_development"
        traces = client.search_traces(experiment_ids=[experiment.experiment_id])
        assert len(traces) == 1
        assert any(span.name == "answer" for span in traces[0].data.spans)
    finally:
        mlflow.langchain.autolog(disable=True)
        mlflow.set_tracking_uri(previous)


def test_missing_remote_endpoint_does_not_fall_back_to_localhost(monkeypatch):
    from stock_assistant.forecast_client import get_models

    monkeypatch.delenv("ASSISTANT_FORECAST_URL", raising=False)
    with patch("stock_assistant.forecast_client.httpx.Client") as client:
        for operation in (lambda: get_prediction("AL"), get_models):
            with pytest.raises(ForecastServiceError, match="No local model fallback"):
                operation()
        client.assert_not_called()


def test_azure_container_apps_al_forecast_preserves_evidence(monkeypatch):
    from stock_assistant.tools.prediction_tools import predict_stock

    base = "https://stock-forecast-api.orangemoss-9dc4289a.uksouth.azurecontainerapps.io"
    version = "AL-lstm-return-h1-2dd84898980a"
    monkeypatch.setenv("ASSISTANT_FORECAST_URL", base)
    monkeypatch.setenv("FORECAST_API_KEY", "test-key")
    payload = {
        **FORECAST,
        "symbol": "AL",
        "model_version": version,
        "deployment_status": "manual_initial_release_approved",
        "approval_id": "fc5e7e9588beb86fb7283fb29347f9d6e8e1cc0f163ec0fa8472230bb64bc1ae",
        "forecast_id": "23824c820926677ea16e2783c25f78de116c7043d1ce2de2da25233bbd0cc4f0",
        "forecast_horizon_sessions": None,
        "horizon_basis": "observed_bars_unverified_sessions",
        "as_of": "2026-10-08",
        "predicted_log_return": 0.001461879350244999,
        "implied_price": 228.33355223901847,
        "price_basis": "unverified",
        "direction": "up",
        "probability_up": None,
        "probability_calibration": None,
        "prediction_interval": None,
        "data_version": "ccf625413d78249f49f0",
        "training_data_version": "129dad463fd1d92f79c0",
        "status": "success",
    }

    def handle(request):
        assert str(request.url) == base + "/v1/predict"
        assert request.headers["x-api-key"] == "test-key"
        assert json.loads(request.content) == {
            "symbol": "AL",
            "model": "lstm",
            "horizon": 1,
            "task": "return",
            "model_version": version,
        }
        return httpx.Response(200, json=payload)

    with patch(
        "stock_assistant.forecast_client.httpx.Client", return_value=httpx.Client(transport=httpx.MockTransport(handle))
    ):
        result = json.loads(predict_stock.invoke({"company": "AL", "model_version": version}))
    assert {key: result[key] for key in payload} == payload
    assert "disclaimer" in result
