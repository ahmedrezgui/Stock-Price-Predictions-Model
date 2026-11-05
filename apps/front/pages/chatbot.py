import streamlit as st
import gc
from time import monotonic
import os
import uuid
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import apply_chat_theme
from chat_stream import stream_response

from dotenv import load_dotenv

load_dotenv()
ASSISTANT_URL = os.environ.get("ASSISTANT_URL", "http://127.0.0.1:8001")

# An agent turn can run several tools and retry a poor answer, so allow for it.
REQUEST_TIMEOUT = 180


GREETING = "Explore a stock, compare performance, or unpack an LSTM forecast. I'll bring together the available data, charts and sources."


def reset_chat():
    st.session_state.active_chat = None
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    # A new thread id starts the assistant on a blank conversation history.
    st.session_state.thread_id = str(uuid.uuid4())
    gc.collect()

def save_chat():
    if st.session_state.active_chat is None and len(st.session_state.messages) > 1:
        st.session_state.counter += 1
        st.session_state.active_chat = {
            "chat_name": next((m["content"][:48] for m in st.session_state.messages if m["role"] == "user"), "Research session"),
            "messages": st.session_state.messages,
            "thread_id": st.session_state.thread_id,
        }
        st.session_state.chats.append(st.session_state.active_chat)


def new_chat():
    save_chat()
    reset_chat()


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            link = source.get("link")
            title = source.get("title") or link or "Untitled"
            entry = f"[{title}]({link})" if link else title
            if source.get("date"):
                entry += f" — {source['date']}"
            st.markdown(f"- {entry}")


def render_charts(charts, key_prefix):
    """
    Draw the figures the assistant produced for an answer.

    The figure arrives as plain JSON built on the server. theme=None keeps the
    colours it was built with, which were chosen to read on either theme; letting
    Streamlit restyle it would override them.

    Args:
        charts: the chart payloads, each with a title, figure and description.
        key_prefix: distinguishes the charts of one message from another's, since
            every chart on the page needs its own key.
    """
    for index, chart in enumerate(charts or []):
        figure = chart.get("figure")
        if not figure:
            continue
        if chart.get("title"):
            st.markdown(f"**{chart['title']}**")
        st.plotly_chart(
            figure,
            use_container_width=True,
            theme=None,
            key=f"{key_prefix}_chart_{index}",
        )
        if chart.get("description"):
            st.caption(chart["description"])


st.set_page_config(page_title="StockWise | Chat", page_icon="S", layout="wide", initial_sidebar_state="expanded")
apply_chat_theme()
st.markdown('<div class="chat-title">Stock<span>Wise</span></div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]

if "chats" not in st.session_state:
    st.session_state.chats = []

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

if "counter" not in st.session_state:
    st.session_state.counter = 0

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ===========================
#   Sidebar
# ===========================
with st.sidebar:
    st.subheader("StockWise")
    st.page_link("home.py", label="Market workspace", icon=":material/candlestick_chart:")
    search = st.text_input("Search conversations", placeholder="Search chats", label_visibility="collapsed")
    st.button("New Chat", on_click=new_chat, icon=":material/edit_square:", use_container_width=True)
    st.caption("Your chats")
    for chat in reversed(list(st.session_state.chats)):
        if search.casefold() not in chat["chat_name"].casefold():
            continue
        cols = st.columns([5, 1])
        selected = st.session_state.thread_id == chat["thread_id"]
        with cols[0]:
            if st.button(chat["chat_name"], key=f"chat_{chat['thread_id']}",
                         type="primary" if selected else "secondary", use_container_width=True, help=chat["chat_name"]):
                save_chat()
                st.session_state.active_chat = chat
                st.session_state.messages = chat["messages"]
                st.session_state.thread_id = chat["thread_id"]
                st.rerun()
        with cols[1]:
            if st.button("", icon=":material/delete:", key=f"delete_{chat['thread_id']}",
                         help="Delete conversation"):
                st.session_state.chats.remove(chat)
                if selected:
                    reset_chat()
                st.rerun()
    st.caption("Chats are saved for this browser session.")

# ===========================
#   Main Chat
# ===========================


for index, message in enumerate(st.session_state.messages):
    if index == 0 and message["content"] == GREETING:
        continue
    with st.chat_message(message["role"]):
        if message.get("activity"):
            with st.expander("Research activity"):
                for step in message["activity"]:
                    st.write(f"{step['label']} - {step['status']}")
        if message.get("elapsed_seconds") is not None:
            st.caption(f"Response time: {message['elapsed_seconds']:.1f}s")
        st.markdown(message["content"])
        render_charts(message.get("charts"), key_prefix=f"history_{index}")
        render_sources(message.get("sources"))


suggested_prompt = None
if len(st.session_state.messages) == 1:
    st.markdown('<h1 class="chat-welcome">What can I help you research?</h1>', unsafe_allow_html=True)
    for column, label, question in zip(st.columns(3),
        ["Explore a forecast", "Compare stocks", "Read the market"],
        ["Plot BIAT and explain its latest deployed LSTM forecast.",
         "Compare BIAT, AL and ATB over their shared recent history, rebased to 100.",
         "What recent news is available for BIAT? Include dates and sources."]):
        if column.button(label, use_container_width=True):
            suggested_prompt = question

prompt = st.chat_input("Message StockWise") or suggested_prompt
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        activity = {}
        res = None
        started = monotonic()
        with st.spinner("Generating response", show_time=True), st.status("Starting research...", expanded=False) as progress:
            activity_placeholder = st.empty()
            try:
                for event in stream_response(ASSISTANT_URL, prompt, st.session_state.thread_id, REQUEST_TIMEOUT):
                    kind = event.get("type")
                    if kind == "status":
                        progress.update(label=event["label"])
                    elif kind == "progress":
                        activity[event["id"]] = event
                        activity_placeholder.markdown("\n\n".join(
                            f"**{step['label']}** - {step['status']}" for step in activity.values()))
                        running = [step for step in activity.values() if step["status"] == "running"]
                        progress.update(label=running[-1]["label"] if running else "Continuing research...")
                    elif kind == "result":
                        res = event
                        failed = any(step["status"] == "failed" for step in activity.values())
                        progress.update(label="Research complete - some tools unavailable" if failed else "Research complete",
                                        state="complete", expanded=False)
                    elif kind == "error":
                        st.error(event["message"])
                        progress.update(label="Research could not finish", state="error", expanded=True)
            except (requests.RequestException, ValueError):
                st.error("The connection to the assistant was interrupted. Please try again shortly.")
                progress.update(label="Connection interrupted", state="error", expanded=True)
            if res is None:
                for step in activity.values():
                    if step["status"] == "running":
                        step["status"] = "interrupted"
                activity_placeholder.markdown("\n\n".join(
                    f"**{step['label']}** - {step['status']}" for step in activity.values()))
        elapsed = monotonic() - started
        st.caption(f"Response time: {elapsed:.1f}s")
        message_placeholder = st.empty()

        if res is not None:
            response = res.get("text", "")
            sources = res.get("sources", [])
            charts = res.get("charts", [])
            message_placeholder.markdown(response)
            render_charts(charts, key_prefix=f"live_{len(st.session_state.messages)}")
            render_sources(sources)

            # Add assistant response to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "sources": sources,
                    "charts": charts,
                    "activity": list(activity.values()),
                    "elapsed_seconds": elapsed,
                }
            )

            save_chat()
            st.rerun()
