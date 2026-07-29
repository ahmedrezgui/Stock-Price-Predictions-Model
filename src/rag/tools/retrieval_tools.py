"""Retrieval tool backed by the Pinecone vector store."""

import json
import os

from langchain_core.tools import tool

from ..pinecone_vector_store import get_pinecone_vector_store

RETRIEVE_NEWS_TOOL = "retrieve_news"
DEFAULT_K = 7

_vector_store = None


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
def retrieve_news(query: str, k: int = DEFAULT_K) -> str:
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
        k: how many articles to return (default 7).

    Returns:
        A JSON object with a "documents" list, each entry holding the article's
        title, date, link and text. An empty list means nothing matched.
    """
    try:
        retriever = get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": {"source": "news"}},
        )
        documents = retriever.invoke(query)
    except Exception as e:
        return json.dumps({"error": f"News search failed: {e}", "documents": []})

    return json.dumps(
        {
            "documents": [
                {
                    "title": d.metadata.get("title"),
                    "date": d.metadata.get("date"),
                    "link": d.metadata.get("link"),
                    "content": d.page_content,
                }
                for d in documents
            ]
        },
        ensure_ascii=False,
        default=str,
    )
