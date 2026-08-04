"""Retrieval tool backed by the Pinecone vector store."""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from langchain_core.tools import tool

from ..agents.document_grader import keep_relevant
from ..pinecone_vector_store import get_pinecone_vector_store

RETRIEVE_NEWS_TOOL = "retrieve_news"
DEFAULT_K = 7

_vector_store = None


def to_timestamp(iso_date: str) -> int:
    """Epoch seconds for a YYYY-MM-DD date, matching what ingestion stores."""
    parsed = datetime.strptime(iso_date, "%Y-%m-%d")
    return int(parsed.replace(tzinfo=timezone.utc).timestamp())


def build_filter(date_from: Optional[str], date_to: Optional[str]) -> dict:
    """
    Build the Pinecone metadata filter for a news search.

    Raises:
        ValueError: a date was not written as YYYY-MM-DD.
    """
    metadata_filter = {"source": "news"}

    window = {}
    if date_from:
        window["$gte"] = to_timestamp(date_from)
    if date_to:
        # Cover the whole of the closing day, not just its first second.
        window["$lte"] = to_timestamp(date_to) + 86399
    if window:
        metadata_filter["timestamp"] = window

    return metadata_filter


def get_vector_store():
    """Return the shared vector store, connecting on first use."""
    global _vector_store
    if _vector_store is None:
        _vector_store = get_pinecone_vector_store(os.getenv("INDEX_NAME"))
    return _vector_store


def reset_vector_store():
    """Forget the cached vector store so the next call reconnects."""
    global _vector_store
    _vector_store = None


@tool(RETRIEVE_NEWS_TOOL)
def retrieve_news(
    query: str,
    k: int = DEFAULT_K,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    """Search the archive of Tunisian financial and stock market news articles.

    Use this for anything about market events, company announcements, economic
    policy, earnings or investor sentiment in Tunisia. Phrase the query as the
    topic you are looking for rather than as the user's literal question, and
    search again with different wording if the first attempt returns nothing
    useful.

    This archive only covers news that has been scraped, so it may not include
    the last day or two. It holds no price history: use get_stock_data for
    numbers.

    Args:
        query: the topic to search for.
        k: how many passages to return (default 7).
        date_from: optional earliest article date, as YYYY-MM-DD.
        date_to: optional latest article date, as YYYY-MM-DD.

    Returns:
        A JSON object with a "documents" list, each entry holding the article's
        title, date, link and text. An empty list means nothing matched; if you
        narrowed the dates, widening them is worth a try.
    """
    try:
        metadata_filter = build_filter(date_from, date_to)
    except ValueError:
        return json.dumps(
            {
                "error": "date_from and date_to must be written as YYYY-MM-DD.",
                "documents": [],
            }
        )

    try:
        retriever = get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": metadata_filter},
        )
        documents = retriever.invoke(query)
    except Exception as e:
        return json.dumps({"error": f"News search failed: {e}", "documents": []})

    passages = [
        {
            "title": d.metadata.get("title"),
            "date": d.metadata.get("date"),
            "link": d.metadata.get("link"),
            "content": d.page_content,
        }
        for d in documents
    ]

    # A similarity search returns its k nearest passages however distant they are,
    # so what comes back is filtered before the agent ever sees it.
    retrieved = len(passages)
    passages = keep_relevant(query, passages)

    payload = {"documents": passages}
    if retrieved and not passages:
        payload["note"] = (
            "The archive returned passages but none of them bore on this query. "
            "Try different wording, widen the dates, or search the web instead."
        )
    elif retrieved > len(passages):
        payload["note"] = (
            f"{retrieved - len(passages)} of {retrieved} retrieved passages were "
            "dropped as unrelated to the query."
        )

    return json.dumps(payload, ensure_ascii=False, default=str)
