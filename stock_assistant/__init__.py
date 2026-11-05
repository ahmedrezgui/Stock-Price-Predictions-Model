"""Stock assistant public entry points; optional integrations load on demand."""

from .graph import ask, create_agents_graph


def get_pinecone_vector_store(*args, **kwargs):
    from .pinecone_vector_store import get_pinecone_vector_store as create

    return create(*args, **kwargs)


__all__ = ["ask", "create_agents_graph"]
