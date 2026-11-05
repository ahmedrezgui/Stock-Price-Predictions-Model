"""Independent dashboard and legacy backend entry points."""


def __getattr__(name):
    if name == "CacheService":
        from .services import CacheService

        return CacheService
    raise AttributeError(name)
