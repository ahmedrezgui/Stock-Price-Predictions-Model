"""Assistant HTTP service. Run separately from the forecast API."""

from contextlib import asynccontextmanager
from threading import Lock
from uuid import uuid4
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.errors import GraphRecursionError
from .graph import ask, create_agents_graph
from .observability import configure_tracking

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=16000)
    thread_id: str = Field(
        default_factory=lambda: str(uuid4()), min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )


def create_app(graph=None):
    @asynccontextmanager
    async def lifespan(app):
        configure_tracking()
        app.state.graph = graph if graph is not None else create_agents_graph()
        yield

    app = FastAPI(title="StockWise assistant", lifespan=lifespan)
    # Striped locks serialize turns for a conversation without unbounded lock growth.
    locks = [Lock() for _ in range(64)]

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "assistant"}

    @app.post("/chat")
    def chat(request: ChatRequest):
        question = request.question.strip()
        if not question:
            raise HTTPException(422, "A question is required")
        try:
            with locks[hash(request.thread_id) % len(locks)]:
                result = ask(app.state.graph, question, request.thread_id)
        except GraphRecursionError:
            raise HTTPException(503, "The assistant reached its tool-call limit. Try a narrower question.") from None
        except Exception:
            logger.exception("Assistant turn failed")
            raise HTTPException(503, "Assistant unavailable. Check provider and service configuration.") from None
        return {**result, "thread_id": request.thread_id}

    return app


app = create_app()
