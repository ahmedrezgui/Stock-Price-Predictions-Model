import streamlit as st
import time
import gc
import torch
import os
import uuid
import requests

from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.environ.get("BACKEND_URL")

# An agent turn can run several tools and retry a poor answer, so allow for it.
REQUEST_TIMEOUT = 180

torch.classes.__path__ = []

GREETING = "Ask questions about Tunisian Stock Market trends 📈"


def get_response(question, thread_id):
    res = requests.post(
        f"{BACKEND_URL}/rag",
        json={"question": question, "thread_id": thread_id},
        timeout=REQUEST_TIMEOUT,
    )
    res.raise_for_status()
    return res.json()

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
            "chat_name": "chat_" + str(st.session_state.counter),
            "messages": st.session_state.messages,
            "thread_id": st.session_state.thread_id,
        }
        st.session_state.chats.append(st.session_state.active_chat)
        st.session_state.active_chat = None


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


st.set_page_config(page_title="StockWise", page_icon="📈", layout="centered")
st.header("💬 AI Chat Assistant")


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
    st.subheader("Chat History")
    for i, chat in enumerate(st.session_state.chats):
        cols = st.columns([4, 1, 1])  # 5:1 ratio between chat name and delete
        with cols[0]:
            if st.button(chat["chat_name"], key=f"chat_{i}"):
                save_chat()
                st.session_state.active_chat = chat
                st.session_state.messages = chat["messages"]
                # Pick the conversation back up where the assistant left it.
                st.session_state.thread_id = chat.get("thread_id", str(uuid.uuid4()))
                gc.collect()
        with cols[1]:
            if st.button("🗑️", key=f"delete_{i}"):
                st.session_state.chats.remove(chat)
                if st.session_state.active_chat == chat:
                    reset_chat()
                gc.collect()

    st.button("New Chat", on_click=new_chat)

# ===========================
#   Main Chat
# ===========================


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources"))


if prompt := st.chat_input("Ask me"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        full_response = ""
        message_placeholder = st.empty()
        with st.spinner("Thinking...", show_time=True):
            try:
                res = get_response(prompt, st.session_state.thread_id)
            except requests.RequestException as e:
                res = None
                st.error(f"The assistant is unreachable right now: {e}")

        if res is not None:
            response = res.get("text", "")
            sources = res.get("sources", [])
            result = response.split(' ')
            for i, word in enumerate(result):
                full_response += word
                if i < len(result) - 1:
                    full_response += ' '

                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.02)

            message_placeholder.markdown(full_response)
            render_sources(sources)

            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": sources}
            )
