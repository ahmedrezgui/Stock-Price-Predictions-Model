"""
Web search, for the questions the stored archive cannot answer.

The news archive only holds what the scraper has collected, so it always trails
the last day or so and covers one publication. This tool fills both gaps. It is
used two ways: the agent reaches for it directly when a question runs past the
archive, and the graph forces one search as a last resort when an answer has been
judged inadequate and the retry budget is spent.

Without TAVILY_API_KEY set, searching reports that it is unavailable rather than
failing: everything else about the assistant still works without it.
"""

import json
import os

from langchain_core.tools import tool

WEB_SEARCH_TOOL = "web_search"

DEFAULT_RESULTS = 5

# The archive is Tunisian market news, so unqualified searches are read the same way.
SEARCH_CONTEXT = "Tunisian stock market and economy"

_client = None


def is_configured() -> bool:
    """Whether a Tavily key is present, and so whether searching can work at all."""
    return bool(os.getenv("TAVILY_API_KEY"))


def get_client():
    """Return the shared Tavily client, creating it on first use."""
    global _client
    if _client is None:
        from tavily import TavilyClient

        _client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _client


def reset_client():
    """Forget the cached client so the next call rebuilds it."""
    global _client
    _client = None


def run_search(query: str, max_results: int = DEFAULT_RESULTS) -> dict:
    """
    Search the web and return results in the same shape as the news archive.

    Keeping the shape identical means the caller collects citations the same way
    whichever source an answer came from.

    Returns:
        A dict with a "documents" list, or an "error" describing why there is none.
    """
    if not is_configured():
        return {
            "error": "Web search is not configured, so only the stored news archive is available.",
            "documents": [],
        }

    try:
        response = get_client().search(
            query=f"{query} {SEARCH_CONTEXT}",
            max_results=max(1, min(max_results, 10)),
            topic="news",
            timeout=30,
            search_depth="basic",
        )
    except Exception as e:
        return {"error": f"Web search failed: {e}", "documents": []}

    return {
        "documents": [
            {
                "title": result.get("title"),
                "date": result.get("published_date"),
                "link": result.get("url"),
                "content": result.get("content"),
            }
            for result in response.get("results", [])
        ]
    }


@tool(WEB_SEARCH_TOOL)
def web_search(query: str, max_results: int = DEFAULT_RESULTS) -> str:
    """Search the open web for Tunisian market information not in the news archive.

    Reach for this when retrieve_news comes back with nothing useful, when the
    question concerns the last day or two that the archive may not cover yet, or
    when it needs a source the archive does not hold. Prefer retrieve_news first:
    it is the curated archive and costs nothing to query.

    Results come from the open web and are not vetted, so attribute what you take
    from them and prefer the archive when the two disagree.

    Args:
        query: what to search for. Tunisian market context is added for you.
        max_results: how many results to return (default 5).

    Returns:
        A JSON object with a "documents" list of title, date, link and extract.
    """
    return json.dumps(run_search(query, max_results), ensure_ascii=False, default=str)
