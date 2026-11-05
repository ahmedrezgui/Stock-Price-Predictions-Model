import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from stock_assistant.api import create_app
from stock_assistant.progress import ProgressHandler, failed_output


def test_progress_never_exposes_inputs_results_or_errors():
    events = []
    handler = ProgressHandler(events.append)
    handler.on_tool_start({"name": "predict_stock"}, "SECRET", run_id="one")
    handler.on_tool_end(ToolMessage(content='{"error":"SECRET"}', tool_call_id="x"), run_id="one")
    handler.on_chain_start({}, {"prompt": "SECRET"}, run_id="two", name="agent")
    handler.on_chain_error(RuntimeError("SECRET"), run_id="two")
    assert [e["status"] for e in events] == ["running", "failed", "running", "failed"]
    assert "SECRET" not in json.dumps(events)
    assert failed_output(Command(update={"messages": [ToolMessage(content='{"error":"bad"}', tool_call_id="x")]}))


def test_stream_real_graph_callbacks_and_final_payload(monkeypatch):
    import stock_assistant.graph as workflow
    monkeypatch.setenv("ASSISTANT_TRACING_ENABLED", "false")
    monkeypatch.setattr(workflow.topic_checker, "invoke", lambda _: type("Topic", (), {"topic": "on_topic"})())
    monkeypatch.setattr(workflow.answer_grader_agent, "invoke", lambda _: type("Grade", (), {"binary_score": "yes"})())
    @tool
    def get_stock_data(company: str):
        """Read test history."""
        return '{"close": 100}'
    llm = MagicMock()
    llm.invoke.side_effect = [AIMessage(content="", tool_calls=[{"name": "get_stock_data", "args": {"company": "BIAT"}, "id": "t1"}]), AIMessage(content="Final answer")]
    monkeypatch.setattr(workflow, "get_assistant_llm", lambda: llm)
    graph = workflow.create_agents_graph(checkpointer=InMemorySaver(), tools=[get_stock_data])
    with TestClient(create_app(graph)) as client:
        response = client.post("/chat/stream", json={"question": "BIAT", "thread_id": "test"})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["text"] == "Final answer"
        assert events[-1]["thread_id"] == "test"
        progress = [e for e in events if e["type"] == "progress"]
        assert any(e["label"] == "Checking your question" for e in progress)
        tools = [e for e in progress if e.get("tool") == "get_stock_data"]
        assert [e["status"] for e in tools] == ["running", "completed"]
        assert tools[0]["id"] == tools[1]["id"]
        assert client.post("/chat/stream", json={"question": " "}).status_code == 422


def test_stream_failure_is_terminal_and_sanitized(monkeypatch):
    monkeypatch.setenv("ASSISTANT_TRACING_ENABLED", "false")
    graph = MagicMock()
    graph.invoke.side_effect = RuntimeError("SECRET")
    with TestClient(create_app(graph)) as client:
        response = client.post("/chat/stream", json={"question": "BIAT"})
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["type"] == "error"
        assert "SECRET" not in response.text


def test_status_arrives_before_graph_finishes(monkeypatch):
    import asyncio
    from threading import Event
    import stock_assistant.api as api
    gate = Event()
    def slow_ask(graph, question, thread_id, callbacks):
        callbacks[0].start("slow", "Loading stock history", "get_stock_data")
        assert gate.wait(5)
        callbacks[0].finish("slow")
        return {"text": "Done", "charts": [], "sources": []}
    monkeypatch.setattr(api, "ask", slow_ask)
    app = create_app(MagicMock())
    app.state.graph = MagicMock()
    endpoint = next(r.endpoint for r in app.routes if r.path == "/chat/stream")
    async def read():
        response = endpoint(api.ChatRequest(question="BIAT"))
        stream = response.body_iterator
        try:
            first = json.loads(await asyncio.wait_for(anext(stream), 2))
            second = json.loads(await asyncio.wait_for(anext(stream), 2))
            assert first["type"] == "status"
            assert second["status"] == "running"
            assert not gate.is_set()
            gate.set()
            remaining = [json.loads(line) async for line in stream]
            assert remaining[-1]["type"] == "result"
        finally:
            gate.set()
            await stream.aclose()
    asyncio.run(read())
