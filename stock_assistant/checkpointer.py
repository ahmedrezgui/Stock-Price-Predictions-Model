"""Conversation memory for the assistant graph."""

import os

try:  # langgraph 1.x renamed the class; older releases only have the old name
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

IN_MEMORY_BACKENDS = {"", "memory", "in_memory", "inmemory"}


def get_checkpointer():
    """
    Build the checkpointer that keeps each conversation's history, selected by the
    CHECKPOINTER_BACKEND environment variable.
    """
    backend = (os.getenv("CHECKPOINTER_BACKEND") or "memory").strip().lower()
    if backend not in IN_MEMORY_BACKENDS:
        # Refusing to start would take the whole API down over a stale value in .env,
        # which is worse than running with the only backend that exists so far.
        print(
            f"Unsupported CHECKPOINTER_BACKEND '{backend}', falling back to in-memory "
            "history. Conversations are lost when the backend restarts."
        )
    return InMemorySaver()
