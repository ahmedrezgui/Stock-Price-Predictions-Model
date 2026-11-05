"""Assistant HTTP service. Run separately from the forecast API."""

from contextlib import asynccontextmanager
from threading import Lock, Thread, Event
from queue import Queue, Empty
import json
from fastapi.responses import StreamingResponse
from .progress import ProgressHandler
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

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest):
        question = request.question.strip()
        if not question:
            raise HTTPException(422, "A question is required")
        events = Queue(maxsize=256)
        disconnected = Event()

        def emit(event):
            # Backpressure bounds memory; disconnect releases a blocked producer.
            from queue import Full
            while not disconnected.is_set():
                try:
                    events.put(event, timeout=0.25)
                    return
                except Full:
                    continue

        def work():
            lock = locks[hash(request.thread_id) % len(locks)]
            try:
                while not lock.acquire(timeout=0.25):
                    if disconnected.is_set():
                        return
                try:
                    if disconnected.is_set():
                        return
                    result = ask(app.state.graph, question, request.thread_id,
                                 callbacks=[ProgressHandler(emit)])
                    emit({"type": "result", **result, "thread_id": request.thread_id})
                finally:
                    lock.release()
            except GraphRecursionError:
                emit({"type": "error", "message": "The assistant reached its tool-call limit. Try a narrower question."})
            except Exception:
                logger.exception("Streaming assistant turn failed")
                emit({"type": "error", "message": "Assistant unavailable. Please try again shortly."})

        def stream():
            Thread(target=work, daemon=True).start()
            try:
                yield json.dumps({"type": "status", "label": "Waiting for the assistant"}) + "\n"
                while True:
                    try:
                        event = events.get(timeout=5)
                    except Empty:
                        yield '{"type":"heartbeat"}\n'
                        continue
                    yield json.dumps(event, ensure_ascii=False) + "\n"
                    if event["type"] in {"result", "error"}:
                        break
            finally:
                disconnected.set()

        return StreamingResponse(stream(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()
