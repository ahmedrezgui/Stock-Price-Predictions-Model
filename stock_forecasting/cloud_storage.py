"""Small identity-authenticated Blob operations for serving and monitoring."""

from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import threading
import time


@lru_cache(maxsize=8)
def container(name):
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    return BlobServiceClient(
        os.environ["AZURE_STORAGE_ACCOUNT_URL"],
        credential=DefaultAzureCredential(),
        retry_total=2,
        connection_timeout=10,
        read_timeout=20,
    ).get_container_client(name)


def read_json(store, key):
    blob = store.get_blob_client(key).download_blob()
    return json.loads(blob.readall()), blob.properties.etag


def write_json(store, key, value, etag=None):
    from azure.core import MatchConditions

    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    options = (
        {"overwrite": True, "etag": etag, "match_condition": MatchConditions.IfNotModified}
        if etag
        else {"overwrite": False}
    )
    return store.upload_blob(key, data, **options)


def immutable(store, key, data):
    from azure.core.exceptions import ResourceExistsError

    try:
        store.upload_blob(key, data, overwrite=False)
    except ResourceExistsError:
        if store.download_blob(key).readall() != data:
            raise ValueError(f"Conflicting immutable object: {key}") from None


def monitoring_ready():
    container(os.environ["FORECAST_MONITOR_CONTAINER"]).get_container_properties()


def save_forecast(payload):
    from azure.core.exceptions import ResourceExistsError

    store = container(os.environ["FORECAST_MONITOR_CONTAINER"])
    key = f"forecasts/{payload['forecast_id']}.json"
    try:
        write_json(store, key, payload)
    except ResourceExistsError:
        saved, _ = read_json(store, key)
        for name in ("data_version", "origin_close", "predicted_log_return", "probability_up", "forecast_for"):
            if saved[name] != payload[name]:
                raise ValueError("Forecast origin/version already exists with different inputs or prediction") from None


def reconcile_remote(data, report, window=30, minimum=20, uri=None):
    """SQLite is an ephemeral calculation workspace; authoritative records remain in Blob."""
    from azure.core.exceptions import ResourceNotFoundError
    from .monitoring import database, run_monitoring

    store = container(os.environ["FORECAST_MONITOR_CONTAINER"])
    versions = {}
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "monitor.sqlite3"
        with database(path) as connection:
            for item in store.list_blobs(name_starts_with="forecasts/"):
                forecast, _ = read_json(store, item.name)
                identifier = forecast["forecast_id"]
                try:
                    previous, etag = read_json(store, f"outcomes/{identifier}.json")
                except ResourceNotFoundError:
                    previous, etag = {}, None
                versions[identifier] = etag
                connection.execute(
                    "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        identifier,
                        identifier,
                        json.dumps(forecast),
                        previous.get("status", "pending"),
                        previous.get("outcome"),
                        previous.get("reason"),
                        previous.get("checked_at"),
                    ),
                )
        result = run_monitoring(path, data, report, window, minimum, uri)
        with database(path) as connection:
            for row in connection.execute("SELECT forecast_id,status,outcome,reason,checked_at FROM forecasts"):
                record = dict(row)
                write_json(store, f"outcomes/{row['forecast_id']}.json", record, versions[row["forecast_id"]])
        # Immutable report history: no last-writer-wins monitoring evidence.
        body = Path(report).read_bytes()
        immutable(store, f"reports/{sha256(body).hexdigest()}.md", body)
        return result


_refresh_lock = threading.Lock()
_refreshed = {}


def refresh_data(root, symbol):
    if os.getenv("FORECAST_DATA_BACKEND", "local") != "azure":
        return
    from .config import Config
    from .data import normalize, read_csv

    root = Path(root)
    key = (str(root.resolve()), symbol)
    with _refresh_lock:
        if time.monotonic() - _refreshed.get(key, -1e20) < 60 and (root / f"{symbol}.csv").exists():
            return
        store = container(os.environ["FORECAST_DATA_CONTAINER"])
        manifest, _ = read_json(store, f"serving/{symbol}.json")
        digest = manifest["sha256"]
        expected = f"snapshots/{symbol}/{digest}.csv"
        if manifest["symbol"] != symbol or manifest["key"] != expected or len(digest) != 64:
            raise ValueError("Invalid serving data manifest")
        data = store.download_blob(expected).readall()
        if sha256(data).hexdigest() != digest:
            raise ValueError("Serving data checksum mismatch")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, suffix=".csv", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        try:
            normalize(read_csv(temporary), Config(symbol=symbol))
            temporary.replace(root / f"{symbol}.csv")
        finally:
            temporary.unlink(missing_ok=True)
        _refreshed[key] = time.monotonic()


def publish_data(source, symbol, container_name):
    import pandas as pd
    from azure.core.exceptions import ResourceNotFoundError
    from .config import Config
    from .data import normalize, read_csv

    frame, _ = normalize(read_csv(source), Config(symbol=symbol))
    if (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - frame.timestamp.iloc[-1]).days > 14:
        raise ValueError("Cannot publish stale serving data")
    data = Path(source).read_bytes()
    digest = sha256(data).hexdigest()
    store = container(container_name)
    key = f"snapshots/{symbol}/{digest}.csv"
    try:
        previous, etag = read_json(store, f"serving/{symbol}.json")
        if previous["last_observation"] > str(frame.timestamp.iloc[-1].date()):
            raise ValueError("Serving data cannot move backwards")
    except ResourceNotFoundError:
        etag = None
    immutable(store, key, data)
    write_json(
        store,
        f"serving/{symbol}.json",
        {"symbol": symbol, "key": key, "sha256": digest, "last_observation": str(frame.timestamp.iloc[-1].date())},
        etag,
    )
    return digest
