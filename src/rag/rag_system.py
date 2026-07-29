"""
The assistant's LangGraph workflow.

A question is first screened for relevance, then handed to a tool-calling agent that
decides for itself what to look up. Once the agent answers, the answer is graded, and a
poor one sends it back round with a bounded retry budget.
"""

import json
import os
import time
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .agents import (
    answer_grader_agent,
    build_system_message,
    get_assistant_llm,
    off_topic_responder,
    topic_checker,
)
from .checkpointer import get_checkpointer
from .tools import AGENT_TOOLS, RETRIEVE_NEWS_TOOL

load_dotenv()

DEFAULT_MAX_RETRIES = 2

RETRY_PROMPT = (
    "That did not fully answer my question. Use your tools to find what is missing "
    "and answer again. If the information simply is not available, say so plainly."
)

FALLBACK_ANSWER = (
    "I could not put together an answer for that. Try rephrasing the question, or ask "
    "about a specific company or a recent market event."
)


class State(TypedDict):
    """
    Attributes:
        messages: the running conversation, including tool calls and their results
        question: the question being answered this turn
        generation: the answer text produced for the user
        sources: the news articles the answer drew on
        retries: how many times the answer has been sent back for another attempt
        grade: whether the last answer was judged to address the question
    """

    messages: Annotated[list, add_messages]
    question: str
    generation: str
    sources: List[dict]
    retries: int
    grade: str


def max_retries() -> int:
    """How many times a poor answer may be retried before we return it anyway."""
    try:
        return max(0, int(os.getenv("AGENT_MAX_RETRIES")))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RETRIES


def log_execution_time(func):
    """
    Decorator to log the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {execution_time:.4f} seconds")
        return result
    return wrapper


def message_text(message) -> str:
    """Flatten a message's content, which Gemini may return as a list of parts."""
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        parts = [
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ]
        content = "".join(parts)
    return content.strip()


def last_ai_message(messages):
    """The agent's most recent reply, wherever it sits in the conversation."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def recent_exchange(messages, limit: int = 4) -> str:
    """
    Summarise the turns before the current question, so that a follow-up like
    "and last month?" is judged on what it follows rather than on its own.
    """
    previous = [
        m
        for m in messages[:-1]
        if isinstance(m, (HumanMessage, AIMessage)) and message_text(m)
    ]
    if not previous:
        return ""
    labelled = [
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {message_text(m)}"
        for m in previous[-limit:]
    ]
    return "\n".join(labelled)


def collect_sources(messages) -> List[dict]:
    """Pull the articles cited by every news search made this turn."""
    sources = []
    seen = set()
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != RETRIEVE_NEWS_TOOL:
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        for document in payload.get("documents", []):
            link = document.get("link")
            if link in seen:
                continue
            seen.add(link)
            sources.append(
                {
                    "title": document.get("title"),
                    "date": document.get("date"),
                    "link": link,
                }
            )
    return sources


@log_execution_time
def check_topic_relevency(state: State):
    """
    Check if the question is worth answering, before any tool is called.

    Args:
        state (State): The current graph state

    Returns:
        str: Next node to call
    """
    question = state["question"]
    history = recent_exchange(state["messages"])
    if history:
        question = f"Conversation so far:\n{history}\n\nNew question: {question}"

    try:
        source = topic_checker.invoke({"question": question})
    except Exception as e:
        # A classifier outage should cost us the guardrail, not the answer.
        print(f"Topic check failed, treating the question as relevant: {e}")
        return "rag"

    return "off_topic" if source.datasource == "off_topic" else "rag"


@log_execution_time
def off_topic(state: State):
    """
    Response for user questions that are off topic.

    Args:
        state (State): The state of the graph.

    Returns:
        state (dict): adds a response to the state indicating the question is off topic.
    """
    response = off_topic_responder.invoke({"question": state["question"]})
    return {
        "generation": response,
        "messages": [AIMessage(content=response)],
        "sources": [],
    }


@log_execution_time
def agent(state: State):
    """
    Let the model answer, calling whichever tools it decides it needs.

    Args:
        state (State): The state of the graph.

    Returns:
        state (dict): the model's reply, which may be a request to run tools.
    """
    response = get_assistant_llm().invoke(
        [build_system_message()] + state["messages"]
    )
    return {"messages": [response]}


def should_continue(state: State):
    """Route the agent to its tools, or on to the answer once it stops asking."""
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else "finalize"


@log_execution_time
def finalize(state: State):
    """
    Lift the agent's reply out of the conversation and gather what it cited.

    Args:
        state (State): The state of the graph.

    Returns:
        state (dict): the answer text and its sources.
    """
    return {
        "generation": message_text(last_ai_message(state["messages"])),
        "sources": collect_sources(state["messages"]),
    }


@log_execution_time
def grade_answer(state: State):
    """
    Judge whether the answer addresses the question, and spend a retry if not.

    Args:
        state (State): The state of the graph.

    Returns:
        state (dict): the grade, plus the retry request when one is affordable.
    """
    generation = state.get("generation") or ""
    retries = state.get("retries") or 0

    if generation:
        try:
            grade = answer_grader_agent.invoke(
                {"question": state["question"], "answer": generation}
            ).binary_score
        except Exception as e:
            # Never withhold a written answer because the grader was unreachable.
            print(f"Answer grading failed, accepting the answer: {e}")
            grade = "yes"
    else:
        grade = "no"

    if grade == "no" and retries < max_retries():
        return {
            "grade": "no",
            "retries": retries + 1,
            "messages": [HumanMessage(content=RETRY_PROMPT)],
        }

    # Out of retries: return the best attempt rather than nothing.
    return {"grade": "yes", "generation": generation or FALLBACK_ANSWER}


def route_after_grade(state: State):
    """Send a rejected answer back to the agent while the retry budget allows."""
    return "retry" if state.get("grade") == "no" else "accept"


def create_agents_graph(checkpointer=None, tools=None):
    """
    Build the assistant graph.

    Args:
        checkpointer: conversation store; defaults to the configured backend.
        tools: the agent's toolbox; defaults to every tool it ships with.

    Returns:
        The compiled graph. Invoking it needs a thread id, for example
        config={"configurable": {"thread_id": "abc"}}.
    """
    print('Creating agents graph...')

    tools = AGENT_TOOLS if tools is None else tools
    checkpointer = get_checkpointer() if checkpointer is None else checkpointer

    workflow = StateGraph(State)

    workflow.add_node("off_topic", off_topic)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("finalize", finalize)
    workflow.add_node("grade_answer", grade_answer)

    workflow.add_conditional_edges(
        START,
        check_topic_relevency,
        {
            "off_topic": "off_topic",
            "rag": "agent",
        },
    )
    workflow.add_edge("off_topic", END)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finalize", "grade_answer")
    workflow.add_conditional_edges(
        "grade_answer",
        route_after_grade,
        {
            "retry": "agent",
            "accept": END,
        },
    )

    return workflow.compile(checkpointer=checkpointer)


def ask(graph, question: str, thread_id: str) -> dict:
    """
    Put one question to the assistant and return its answer with any sources.

    Args:
        graph: a graph built by create_agents_graph.
        question: the user's question.
        thread_id: identifies the conversation this question belongs to.
    """
    state = graph.invoke(
        {"question": question, "messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return {
        "text": state.get("generation") or FALLBACK_ANSWER,
        "sources": state.get("sources") or [],
    }
