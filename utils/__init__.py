"""Legacy utilities, loaded only when requested."""
from importlib import import_module
from .constants import HEADERS, SYMBOLS, RAW_DATA_DOWNLOAD_BASELINK, STOCK_DATA_URL, NEWS_BASE_URL, PAGE_URL


def __getattr__(name):
    module = ".email_utils" if name == "send_email" else ".train_test_utils"
    return getattr(import_module(module, __name__), name)
