import json
import os
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from stock_assistant.agents.topic_check import TopicCheck
from stock_assistant.checkpointer import InMemorySaver
from stock_assistant.graph import (
    FALLBACK_ANSWER,
    collect_sources,
    create_agents_graph,
    max_retries,
    message_text,
    recent_exchange,
    route_after_grade,
    should_continue,
)


class ScriptedLLM:
    """Stands in for the tool-bound chat model, replying from a fixed script."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        # A real model returns a fresh message each call; reusing one would collide
        # on id and overwrite the earlier reply instead of appending to it.
        return response.model_copy(update={"id": str(uuid4())})


def tool_call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def graded(*verdicts):
    """A grader that returns each verdict in turn, then repeats the last."""
    remaining = list(verdicts)

    def invoke(_payload):
        verdict = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return MagicMock(binary_score=verdict)

    grader = MagicMock()
    grader.invoke.side_effect = invoke
    return grader


def relevance(verdict):
    checker = MagicMock()
    checker.invoke.return_value = TopicCheck(topic=verdict)
    return checker


def build(llm, topic="on_topic", grades=("yes",), tools=None, web_results=None):
    """
    Compile a graph wired to scripted stand-ins for every model call.

    Args:
        web_results: when given, the fallback search is treated as configured and
            returns these documents; otherwise it is treated as unavailable.
    """
    patches = [
        patch("stock_assistant.graph.get_assistant_llm", return_value=llm),
        patch("stock_assistant.graph.topic_checker", relevance(topic)),
        patch("stock_assistant.graph.answer_grader_agent", graded(*grades)),
        patch(
            "stock_assistant.graph.web_search_configured",
            return_value=web_results is not None,
        ),
        patch(
            "stock_assistant.graph.run_search",
            return_value={"documents": web_results or []},
        ),
    ]
    for p in patches:
        p.start()
    graph = create_agents_graph(checkpointer=InMemorySaver(), tools=tools or [])
    return graph, patches


def ask(graph, question, thread_id="t1"):
    return graph.invoke(
        {"question": question, "messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )


# ------------------------------------------------------------------- pure helpers


def test_message_text_flattens_multi_part_content():
    assert message_text(AIMessage(content=[{"text": "a "}, {"text": "b"}])) == "a b"
    assert message_text(AIMessage(content="  hi  ")) == "hi"
    assert message_text(AIMessage(content="")) == ""


def test_recent_exchange_skips_the_current_question():
    messages = [
        HumanMessage(content="How did BIAT do?"),
        AIMessage(content="It rose 2%."),
        HumanMessage(content="and last month?"),
    ]

    summary = recent_exchange(messages)

    assert "How did BIAT do?" in summary
    assert "It rose 2%." in summary
    assert "and last month?" not in summary


def test_recent_exchange_is_empty_on_the_first_turn():
    assert recent_exchange([HumanMessage(content="hello")]) == ""


def test_collect_sources_dedupes_by_link():
    payload = json.dumps(
        {
            "documents": [
                {"title": "A", "date": "01/03/2024", "link": "https://x/a"},
                {"title": "A again", "date": "01/03/2024", "link": "https://x/a"},
                {"title": "B", "date": "02/03/2024", "link": "https://x/b"},
            ]
        }
    )
    messages = [ToolMessage(content=payload, name="retrieve_news", tool_call_id="1")]

    assert collect_sources(messages) == [
        {"title": "A", "date": "01/03/2024", "link": "https://x/a"},
        {"title": "B", "date": "02/03/2024", "link": "https://x/b"},
    ]


def test_collect_sources_ignores_other_tools_and_bad_payloads():
    messages = [
        ToolMessage(content="not json", name="retrieve_news", tool_call_id="1"),
        ToolMessage(content='{"rows": []}', name="get_stock_data", tool_call_id="2"),
    ]

    assert collect_sources(messages) == []


def test_should_continue_follows_the_tool_calls():
    with_calls = {"messages": [tool_call("retrieve_news", {"query": "x"})]}
    without = {"messages": [AIMessage(content="done")]}

    assert should_continue(with_calls) == "tools"
    assert should_continue(without) == "finalize"


def test_route_after_grade():
    assert route_after_grade({"grade": "no"}) == "retry"
    assert route_after_grade({"grade": "yes"}) == "accept"
    assert route_after_grade({}) == "accept"


@patch.dict(os.environ, {"AGENT_MAX_RETRIES": "3"})
def test_max_retries_reads_the_environment():
    assert max_retries() == 3


@patch.dict(os.environ, {"AGENT_MAX_RETRIES": "not a number"})
def test_max_retries_falls_back_when_misconfigured():
    assert max_retries() == 2


# ------------------------------------------------------------------------- graph


def test_topic_check_classifies_relevance_instead_of_selecting_retrieval():
    assert TopicCheck(topic="on_topic").topic == "on_topic"
    assert TopicCheck(topic="off_topic").topic == "off_topic"
    with pytest.raises(ValidationError):
        TopicCheck(topic="rag")


def test_off_topic_question_never_reaches_the_agent():
    llm = ScriptedLLM(AIMessage(content="should not be used"))
    graph, patches = build(llm, topic="off_topic")
    try:
        with patch("stock_assistant.graph.off_topic_responder") as responder:
            responder.invoke.return_value = "I only answer stock market questions."
            state = ask(graph, "who won the match?")
    finally:
        for p in patches:
            p.stop()

    assert state["generation"] == "I only answer stock market questions."
    assert state["sources"] == []
    assert llm.calls == []


def test_a_good_answer_ends_the_turn():
    llm = ScriptedLLM(AIMessage(content="BIAT closed at 100."))
    graph, patches = build(llm)
    try:
        with patch("stock_assistant.tools.retrieval_tools.get_vector_store") as store:
            state = ask(graph, "what did BIAT close at?")
            store.assert_not_called()
    finally:
        for p in patches:
            p.stop()

    assert state["generation"] == "BIAT closed at 100."
    assert len(llm.calls) == 1


def test_the_agent_runs_a_tool_and_its_sources_are_collected():
    documents = json.dumps(
        {
            "documents": [
                {"title": "Market up", "date": "01/03/2024", "link": "https://x/a", "content": "The market rose."}
            ]
        }
    )

    llm = ScriptedLLM(
        tool_call("retrieve_news", {"query": "market"}),
        AIMessage(content="The market rose last week."),
    )
    with patch("stock_assistant.tools.retrieval_tools.get_vector_store") as store:
        retriever = MagicMock()
        retriever.invoke.return_value = []
        store.return_value.as_retriever.return_value = retriever

        from stock_assistant.tools import retrieve_news

        graph, patches = build(llm, tools=[retrieve_news])
        try:
            with patch.object(retrieve_news, "func", return_value=documents):
                state = ask(graph, "what happened in the market?")
        finally:
            for p in patches:
                p.stop()

    assert state["generation"] == "The market rose last week."
    assert state["sources"] == [{"title": "Market up", "date": "01/03/2024", "link": "https://x/a"}]
    # system + question, then system + question + tool call + tool result
    assert len(llm.calls) == 2


@patch.dict(os.environ, {"AGENT_MAX_RETRIES": "2"})
def test_a_rejected_answer_is_not_published_after_retries():
    llm = ScriptedLLM(AIMessage(content="a weak answer"))
    graph, patches = build(llm, grades=("no",))
    try:
        state = ask(graph, "why did the market move?")
    finally:
        for p in patches:
            p.stop()

    # The loop must stop rather than spin: first attempt plus two retries.
    assert len(llm.calls) == 3
    assert state["retries"] == 2
    assert state["generation"] == FALLBACK_ANSWER


@patch.dict(os.environ, {"AGENT_MAX_RETRIES": "2"})
def test_a_retry_can_still_produce_a_good_answer():
    llm = ScriptedLLM(
        AIMessage(content="a weak answer"),
        AIMessage(content="a better answer"),
    )
    graph, patches = build(llm, grades=("no", "yes"))
    try:
        state = ask(graph, "why did the market move?")
    finally:
        for p in patches:
            p.stop()

    assert state["generation"] == "a better answer"
    assert state["retries"] == 1


def test_an_empty_answer_falls_back_to_a_readable_message():
    llm = ScriptedLLM(AIMessage(content=""))
    graph, patches = build(llm, grades=("no",))
    try:
        with patch.dict(os.environ, {"AGENT_MAX_RETRIES": "0"}):
            state = ask(graph, "why did the market move?")
    finally:
        for p in patches:
            p.stop()

    assert state["generation"] == FALLBACK_ANSWER


def test_the_conversation_is_remembered_across_turns():
    llm = ScriptedLLM(AIMessage(content="BIAT closed at 100."))
    graph, patches = build(llm)
    try:
        ask(graph, "what did BIAT close at?", thread_id="thread-a")
        ask(graph, "and last month?", thread_id="thread-a")
    finally:
        for p in patches:
            p.stop()

    # The second turn is sent the first exchange, not just the new question.
    second_turn = llm.calls[-1]
    assert any(isinstance(m, HumanMessage) and m.content == "what did BIAT close at?" for m in second_turn)


@patch.dict(os.environ, {"AGENT_MAX_RETRIES": "1"})
def test_the_web_is_searched_once_before_giving_up():
    llm = ScriptedLLM(AIMessage(content="a weak answer"))
    graph, patches = build(
        llm,
        grades=("no",),
        web_results=[
            {
                "title": "Late news",
                "date": "2024-03-01",
                "link": "https://x/late",
                "content": "Something happened.",
            }
        ],
    )
    try:
        state = ask(graph, "what happened yesterday?")
    finally:
        for p in patches:
            p.stop()

    assert state["web_searched"] is True
    # First attempt, one retry, then one more after the search.
    assert len(llm.calls) == 3
    # What the search turned up is citable even though no tool call produced it.
    assert state["sources"] == [{"title": "Late news", "date": "2024-03-01", "link": "https://x/late"}]


@patch.dict(os.environ, {"AGENT_MAX_RETRIES": "1"})
def test_the_web_is_not_searched_when_it_is_unconfigured():
    llm = ScriptedLLM(AIMessage(content="a weak answer"))
    graph, patches = build(llm, grades=("no",), web_results=None)
    try:
        state = ask(graph, "what happened yesterday?")
    finally:
        for p in patches:
            p.stop()

    assert not state.get("web_searched")
    assert len(llm.calls) == 2
    assert state["generation"] == FALLBACK_ANSWER


@patch.dict(os.environ, {"AGENT_MAX_RETRIES": "0"})
def test_the_fallback_search_is_spent_only_once():
    llm = ScriptedLLM(AIMessage(content="a weak answer"))
    graph, patches = build(llm, grades=("no",), web_results=[])
    try:
        state = ask(graph, "what happened yesterday?")
    finally:
        for p in patches:
            p.stop()

    # Without this the graph would search, fail, and search again forever.
    assert len(llm.calls) == 2
    assert state["generation"] == FALLBACK_ANSWER


def test_a_chart_reaches_the_answer():
    from stock_assistant.tools import plot_stock

    llm = ScriptedLLM(
        tool_call("plot_stock", {"company": "AB"}),
        AIMessage(content="Here is the chart."),
    )
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "ouverture": [1.0, 2.0, 3.0],
            "haut": [1.0, 2.0, 3.0],
            "bas": [1.0, 2.0, 3.0],
            "cloture": [1.0, 2.0, 3.0],
            "volume": [10, 10, 10],
        }
    )

    graph, patches = build(llm, tools=[plot_stock])
    try:
        with patch("stock_assistant.tools.frames.load_company_df", return_value=history):
            state = ask(graph, "chart AB")
    finally:
        for p in patches:
            p.stop()

    assert len(state["charts"]) == 1
    assert state["charts"][0]["title"] == "AB closing price"


def test_a_chart_does_not_follow_the_conversation_into_the_next_answer():
    from stock_assistant.tools import plot_stock

    llm = ScriptedLLM(
        tool_call("plot_stock", {"company": "AB"}),
        AIMessage(content="Here is the chart."),
        AIMessage(content="No chart this time."),
    )
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "ouverture": [1.0, 2.0, 3.0],
            "haut": [1.0, 2.0, 3.0],
            "bas": [1.0, 2.0, 3.0],
            "cloture": [1.0, 2.0, 3.0],
            "volume": [10, 10, 10],
        }
    )

    graph, patches = build(llm, tools=[plot_stock])
    try:
        with patch("stock_assistant.tools.frames.load_company_df", return_value=history):
            first = ask(graph, "chart AB", thread_id="thread-c")
            second = ask(graph, "and what about the news?", thread_id="thread-c")
    finally:
        for p in patches:
            p.stop()

    assert len(first["charts"]) == 1
    assert second["charts"] == []


@patch.dict(os.environ, {"AGENT_MAX_RETRIES": "1"})
def test_the_retry_budget_is_restored_for_each_new_question():
    llm = ScriptedLLM(AIMessage(content="a weak answer"))
    graph, patches = build(llm, grades=("no",))
    try:
        ask(graph, "why did the market move?", thread_id="thread-d")
        ask(graph, "and last month?", thread_id="thread-d")
    finally:
        for p in patches:
            p.stop()

    # One attempt plus one retry, twice over. Without a per-turn reset the second
    # question would inherit the spent budget and never get its retry.
    assert len(llm.calls) == 4


def test_separate_threads_do_not_share_history():
    llm = ScriptedLLM(AIMessage(content="BIAT closed at 100."))
    graph, patches = build(llm)
    try:
        ask(graph, "what did BIAT close at?", thread_id="thread-a")
        ask(graph, "and last month?", thread_id="thread-b")
    finally:
        for p in patches:
            p.stop()

    second_turn = llm.calls[-1]
    assert not any(isinstance(m, HumanMessage) and m.content == "what did BIAT close at?" for m in second_turn)


def test_failed_news_fallback_keeps_current_stock_request_in_focus():
    question = "Plot AL stock and explain its LSTM forecast with recent news"
    llm = ScriptedLLM(AIMessage(content="Here is information about thermal energy."))
    graph, patches = build(llm, grades=("no",), web_results=[])
    try:
        with patch.dict(os.environ, {"AGENT_MAX_RETRIES": "0"}):
            state = ask(graph, question)
    finally:
        for p in patches:
            p.stop()
    assert state["generation"] == FALLBACK_ANSWER
    assert state["messages"][-1].content == FALLBACK_ANSWER
    assert question in llm.calls[-1][-1].content
    assert any(question in m.content for m in llm.calls[-1] if m.type == "system")
