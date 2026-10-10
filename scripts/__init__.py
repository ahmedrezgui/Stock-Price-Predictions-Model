"""Command entry points; importing this package has no ETL side effects."""
from importlib import import_module


def __getattr__(name):
    modules = {"get_dates": "download_data", "update_dates": "download_data", "download": "download_data",
               "process_all_raw_files": "process_data", "fill_missing_dates_interpolation": "process_data"}
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f".{modules[name]}", __name__), name)
