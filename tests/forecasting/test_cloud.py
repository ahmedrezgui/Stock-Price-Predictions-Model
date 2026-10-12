from types import SimpleNamespace

import pytest
from azure.core.exceptions import ResourceExistsError, ResourceModifiedError, ResourceNotFoundError

from stock_forecasting import cloud_storage
from stock_forecasting.azure_registry import AzureRegistry
from stock_forecasting.config import Config
from stock_forecasting.monitoring import save_forecast


class BlobMemory:
    def __init__(self):
        self.objects = {}

    def upload_blob(self, name, data, overwrite=False, etag=None, **kwargs):
        previous = self.objects.get(name)
        if previous and not overwrite:
            raise ResourceExistsError("exists")
        if etag and (not previous or previous[1] != etag):
            raise ResourceModifiedError("changed")
        self.objects[name] = (data, str(int(previous[1]) + 1) if previous else "1")

    def get_blob_client(self, name):
        return SimpleNamespace(download_blob=lambda: self.download_blob(name))

    def download_blob(self, name):
        if name not in self.objects:
            raise ResourceNotFoundError("missing")
        data, etag = self.objects[name]
        return SimpleNamespace(readall=lambda: data, properties=SimpleNamespace(etag=etag))

    def list_blobs(self, name_starts_with):
        return [SimpleNamespace(name=k) for k in self.objects if k.startswith(name_starts_with)]


@pytest.fixture
def remote(monkeypatch):
    store = BlobMemory()
    monkeypatch.setattr(cloud_storage, "container", lambda name: store)
    monkeypatch.setenv("FORECAST_MONITOR_BACKEND", "azure")
    monkeypatch.setenv("FORECAST_MONITOR_CONTAINER", "test-monitor")
    return store


def test_remote_forecasts_survive_local_cache_and_reconcile(remote, tmp_path, bars, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    forecast = {
        "symbol": "TEST",
        "model": "lstm",
        "model_version": "v1",
        "price_basis": "raw",
        "as_of": str(bars.timestamp.iloc[10].date()),
        "forecast_for": None,
        "horizon_basis": "observed_bars_unverified_sessions",
        "forecast_horizon_observations": 1,
        "predicted_log_return": 0.01,
        "probability_up": None,
        "direction": "up",
        "data_version": "v1",
    }
    config = Config(symbol="TEST", adjustment="raw")
    first = save_forecast(tmp_path / "unused.db", forecast, config, bars.close.iloc[10], now="2020-01-16")
    second = save_forecast(tmp_path / "another.db", forecast, config, bars.close.iloc[10], now="2020-01-17")
    assert first == second and len(remote.objects) == 1
    assert not (tmp_path / "unused.db").exists()
    with pytest.raises(ValueError, match="different"):
        save_forecast(
            tmp_path / "unused.db",
            {**forecast, "predicted_log_return": 0.05},
            config,
            bars.close.iloc[10],
            now="2020-01-16",
        )
    bars.to_csv(tmp_path / "TEST.csv", index=False)
    result = cloud_storage.reconcile_remote(tmp_path, tmp_path / "report.md")
    assert result["groups"][0]["latest"]["n"] == 1
    record, _ = cloud_storage.read_json(remote, f"outcomes/{first}.json")
    assert record["status"] == "matched"
    assert cloud_storage.reconcile_remote(tmp_path, tmp_path / "report.md")["forecast_count"] == 1


def test_data_refresh_checks_hash_and_replaces_cache(remote, tmp_path, bars, monkeypatch):
    import pandas as pd
    from hashlib import sha256

    monkeypatch.setenv("FORECAST_DATA_BACKEND", "azure")
    monkeypatch.setenv("FORECAST_DATA_CONTAINER", "test-data")
    bars.timestamp = pd.bdate_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=len(bars))
    source = tmp_path / "TEST.csv"
    bars.to_csv(source, index=False)
    digest = cloud_storage.publish_data(source, "TEST", "test-data")
    cache = tmp_path / "cache"
    cloud_storage.refresh_data(cache, "TEST")
    assert sha256((cache / "TEST.csv").read_bytes()).hexdigest() == digest
    cloud_storage._refreshed.clear()
    remote.objects[f"snapshots/TEST/{digest}.csv"] = (b"corrupt", "2")
    with pytest.raises(ValueError, match="checksum"):
        cloud_storage.refresh_data(cache, "TEST")
    assert sha256((cache / "TEST.csv").read_bytes()).hexdigest() == digest


def test_publication_uses_same_whole_day_freshness_limit_as_api(remote, tmp_path, bars):
    import pandas as pd

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    bars.timestamp = pd.date_range(end=today - pd.Timedelta(days=14), periods=len(bars))
    source = tmp_path / "TEST.csv"
    bars.to_csv(source, index=False)
    assert cloud_storage.publish_data(source, "TEST", "test-data")
    bars.timestamp -= pd.Timedelta(days=1)
    bars.to_csv(source, index=False)
    with pytest.raises(ValueError, match="stale"):
        cloud_storage.publish_data(source, "TEST", "test-data")


def test_azure_registry_state_is_conditional_and_preview_cannot_promote(remote):
    client = AzureRegistry.__new__(AzureRegistry)
    client.state = remote
    client.set_registered_model_alias("stock-TEST-return-h1", "candidate", "v1")
    assert not client.get_registered_model("stock-TEST-return-h1").aliases.get("deployed")
    with pytest.raises(ValueError, match="confirmation"):
        client.set_registered_model_alias("stock-TEST-return-h1", "deployed", "v1")
    client.confirm("stock-TEST-return-h1", "v1", None, "https://example.test", "gate1")
    with pytest.raises(ValueError, match="changed"):
        client.confirm("stock-TEST-return-h1", "v2", None, "https://example.test", "gate2")
    value, etag = cloud_storage.read_json(remote, "registry/stock-TEST-return-h1.json")
    cloud_storage.write_json(remote, "registry/stock-TEST-return-h1.json", value, etag)
    with pytest.raises(ResourceModifiedError):
        cloud_storage.write_json(remote, "registry/stock-TEST-return-h1.json", value, etag)


def test_initial_authorization_is_one_time_and_never_replaces_incumbent(remote, tmp_path):
    import json
    from stock_forecasting.initial_release import reserve

    client = AzureRegistry.__new__(AzureRegistry)
    client.state = remote
    client.get_model_version = lambda name, version: SimpleNamespace(tags={"bundle_sha256": "weights"})
    approval = {
        "scope": "initial_release_only",
        "approved_by": "owner",
        "reason": "Explicit first release",
        "production_gate_passed": False,
        "models": [{"registry_name": "stock-TEST-return-h1", "registry_version": "1", "bundle_sha256": "weights"}],
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(approval))
    _, digest = reserve(client, path)
    assert reserve(client, path)[1] == digest  # Retry of the exact same authorization.
    assert client.get_registered_model("stock-TEST-return-h1").aliases == {}
    changed = {**approval, "reason": "Another purported initial release"}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Conflicting immutable"):
        reserve(client, path)
    path.write_text(json.dumps(approval))
    client.confirm("stock-TEST-return-h1", "2", None, "https://example.test", "gate")
    with pytest.raises(ValueError, match="replacements require"):
        reserve(client, path)
    approval["production_gate_passed"] = True
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="must not claim"):
        reserve(client, path)


def test_replacement_preserves_other_stocks_and_detects_concurrent_change(tmp_path, monkeypatch):
    import json
    from stock_forecasting import portfolio

    output = tmp_path / "deployment"
    (output / "models").mkdir(parents=True)
    (output / "data").mkdir()
    (output / "models/active.json").write_text(json.dumps({"active": {"NEW-lstm-return-h1": "new-v2"}}))
    old = tmp_path / "old-v1"
    old.mkdir()
    (old / "weights.pt").write_bytes(b"retained weights")
    roster = tmp_path / "roster.json"
    roster.write_text(
        json.dumps({"models": [{"registry_name": "stock-NEW-return-h1"}, {"registry_name": "stock-OLD-return-h1"}]})
    )
    current = SimpleNamespace(
        aliases={"deployed": "1"},
        tags={"deployment_approval": "manual_initial_release_approved", "initial_approval_sha256": "owner-proof"},
    )
    client = SimpleNamespace(get_registered_model=lambda name: current)
    model = SimpleNamespace(config=Config(symbol="OLD"))
    monkeypatch.setattr(
        portfolio, "registered_bundle", lambda *args: (old, model, None, {"model_version": "old-v1"}, None)
    )
    monkeypatch.setattr(
        portfolio, "refresh_data", lambda root, symbol: (root / f"{symbol}.csv").write_text("retained data")
    )
    portfolio.retain_deployed(client, roster, output, "stock-NEW-return-h1")
    state = json.loads((output / "models/active.json").read_text())
    assert state["active"] == {"NEW-lstm-return-h1": "new-v2", "OLD-lstm-return-h1": "old-v1"}
    assert state["approvals"]["OLD-lstm-return-h1"]["approval_id"] == "owner-proof"
    portfolio.verify_retained(client, output)
    current.aliases["deployed"] = "2"
    with pytest.raises(ValueError, match="portfolio member changed"):
        portfolio.verify_retained(client, output)
