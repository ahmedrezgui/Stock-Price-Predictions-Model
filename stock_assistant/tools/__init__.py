from .analytics_tools import analyze_stock_performance, compare_stocks, top_movers
from .plot_tools import plot_comparison, plot_stock, plot_prediction
from .prediction_tools import predict_stock, list_forecast_models
from .retrieval_tools import (
    RETRIEVE_NEWS_TOOL,
    get_vector_store,
    reset_vector_store,
    retrieve_news,
)
from .stock_data_tools import get_stock_data, list_available_companies
from .web_search_tools import WEB_SEARCH_TOOL, is_configured, run_search, web_search

# The toolbox handed to the agent, grouped by what each group is for: finding out
# what happened, reading the numbers, working them out, drawing them, and looking
# beyond the archive.
AGENT_TOOLS = [
    retrieve_news,
    get_stock_data,
    predict_stock,
    list_forecast_models,
    analyze_stock_performance,
    compare_stocks,
    top_movers,
    plot_stock,
    plot_prediction,
    plot_comparison,
    web_search,
    list_available_companies,
]

# Tools whose results are citable, and so are collected as sources for the answer.
SOURCE_TOOLS = {RETRIEVE_NEWS_TOOL, WEB_SEARCH_TOOL}

__all__ = [
    "analyze_stock_performance",
    "compare_stocks",
    "top_movers",
    "plot_comparison",
    "plot_stock",
    "plot_prediction",
    "predict_stock",
    "list_forecast_models",
    "RETRIEVE_NEWS_TOOL",
    "get_vector_store",
    "reset_vector_store",
    "retrieve_news",
    "get_stock_data",
    "list_available_companies",
    "WEB_SEARCH_TOOL",
    "is_configured",
    "run_search",
    "web_search",
]
