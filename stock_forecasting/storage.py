"""Immutable storage: no connection keys, no silent historical overwrites."""

from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
from typing import Protocol

import pandas as pd


class Store(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...


def validate_key(key: str):
    if not key or PurePosixPath(key).is_absolute() or ".." in PurePosixPath(key).parts or "\\" in key or ":" in key:
        raise ValueError("Storage key must be a relative POSIX path without traversal")


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def path(self, key: str) -> Path:
        validate_key(key)
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Storage key escapes root")
        return path

    def put(self, key: str, data: bytes) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(data)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FileExistsError(f"Immutable version already exists: {key}") from None

    def get(self, key: str) -> bytes:
        return self.path(key).read_bytes()


class AzureStore:
    def __init__(self, account_url: str, container: str, client=None):
        if not account_url.startswith("https://") or not container:
            raise ValueError("HTTPS account URL and existing container required")
        if client is None:
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient

            client = BlobServiceClient(account_url, credential=DefaultAzureCredential())
        self.container = client.get_container_client(container)

    def put(self, key: str, data: bytes) -> None:
        from azure.core.exceptions import ResourceExistsError

        validate_key(key)
        try:
            self.container.upload_blob(name=key, data=data, overwrite=False)
        except ResourceExistsError:
            if self.get(key) != data:
                raise FileExistsError(f"Immutable blob already exists: {key}") from None

    def get(self, key: str) -> bytes:
        validate_key(key)
        return self.container.download_blob(key).readall()


def configured_store() -> Store:
    backend = os.getenv("FORECAST_STORAGE", "local")
    if backend == "local":
        return LocalStore(os.getenv("FORECAST_STORAGE_ROOT", "artifacts/storage"))
    if backend == "azure":
        return AzureStore(os.environ["AZURE_STORAGE_ACCOUNT_URL"], os.environ["AZURE_STORAGE_CONTAINER"])
    raise ValueError("FORECAST_STORAGE must be local or azure")


def snapshot_identity(frame: pd.DataFrame, metadata: dict):
    """Compute the same dataset identity with or without persisting a snapshot."""
    stream = BytesIO()
    frame.to_parquet(stream, index=False)
    data = stream.getvalue()
    checksum = sha256(data).hexdigest()
    identity = json.dumps(metadata, sort_keys=True).encode() + data
    version = sha256(identity).hexdigest()[:20]
    key = f"market-data/processed/{version}/{metadata['symbol']}.parquet"
    manifest = {**metadata, "dataset_version": version, "sha256": checksum, "key": key}
    return manifest, data


def snapshot(frame: pd.DataFrame, metadata: dict, store: Store) -> dict:
    manifest, data = snapshot_identity(frame, metadata)
    key, version = manifest["key"], manifest["dataset_version"]
    store.put(key, data)
    store.put(f"market-data/processed/{version}/manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode())
    return manifest


def retrieve(manifest: dict, store: Store) -> pd.DataFrame:
    data = store.get(manifest["key"])
    if sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError("Dataset checksum mismatch")
    return pd.read_parquet(BytesIO(data))
