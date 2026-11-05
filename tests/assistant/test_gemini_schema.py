"""Exercise request serialization, which bind_tools alone defers until invoke."""

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from stock_assistant.tools import AGENT_TOOLS, predict_stock


def test_gemini_serializes_every_tool_before_sending():
    integration = pytest.importorskip("langchain_google_genai")
    from google.ai.generativelanguage import GenerateContentResponse

    model = integration.ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="offline-placeholder")
    response = GenerateContentResponse(
        candidates=[{"content": {"role": "model", "parts": [{"text": "Ready"}]}, "finish_reason": "STOP"}]
    )
    # Mock only the network boundary so the real adapter constructs its protobuf request.
    with patch.object(model.client, "generate_content", return_value=response) as send:
        result = model.bind_tools(AGENT_TOOLS).invoke([HumanMessage(content="Plot BIAT")])
    assert result.content == "Ready"
    request = send.call_args.kwargs["request"]
    declarations = [declaration for entry in request.tools for declaration in entry.function_declarations]
    assert {declaration.name for declaration in declarations} == {tool.name for tool in AGENT_TOOLS}
    prediction = next(declaration for declaration in declarations if declaration.name == "predict_stock")
    assert not prediction.parameters.properties["horizon"].enum


@pytest.mark.parametrize("horizon", [0, 2, -1, 21])
def test_invalid_tool_horizon_is_rejected_before_http(horizon):
    with patch("stock_assistant.forecast_client.httpx.Client") as client:
        result = json.loads(predict_stock.invoke({"company": "BIAT", "horizon": horizon}))
    assert "error" in result
    client.assert_not_called()
