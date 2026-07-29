from .prediction_tools import predict_stock
from .retrieval_tools import (
    RETRIEVE_NEWS_TOOL,
    get_vector_store,
    reset_vector_store,
    retrieve_news,
)
from .stock_data_tools import get_stock_data, list_available_companies

# The toolbox handed to the agent, ordered roughly by how often it is needed.
AGENT_TOOLS = [
    retrieve_news,
    get_stock_data,
    predict_stock,
    list_available_companies,
]
