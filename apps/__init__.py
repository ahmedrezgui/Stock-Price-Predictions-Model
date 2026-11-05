"""Application entry points, without importing optional legacy services."""


def __getattr__(name):
    if name == "CacheService":
        from .back.services import CacheService

        return CacheService
    raise AttributeError(name)
