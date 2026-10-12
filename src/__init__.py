"""Public imports loaded on demand; forecasting needs no cloud or agent clients."""
from importlib import import_module

_EXPORTS = {
    "CacheService": ".web", "IModel": ".prediction_model", "ARIMAModel": ".prediction_model",
    "LSTMModel": ".prediction_model", "GRUModel": ".prediction_model", "create_model": ".factories",
    **{name: ".handlers" for name in ("get_or_create_scaler", "get_or_create_model", "save_scaler", "save_model", "get_model", "get_scaler")},
}

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value

def get_pinecone_vector_store(*args, **kwargs):
    from .rag import get_pinecone_vector_store as create_store
    return create_store(*args, **kwargs)
