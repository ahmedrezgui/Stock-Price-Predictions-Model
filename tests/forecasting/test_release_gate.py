import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.config import Config
from stock_forecasting import deployment, release_gate as gate, release
from stock_forecasting.model_registry import deployed_version, registered_bundle
from stock_forecasting.training import train_candidate

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def case(bars, tmp_path, monkeypatch):
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "gate-test")
    client = mlflow.MlflowClient(tracking_uri=uri)
    client.create_experiment("gate-test", artifact_location=(tmp_path / "mlruns").as_uri())
    start = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) + pd.Timedelta(days=1)
    bars.timestamp = pd.bdate_range(start, periods=len(bars))
    now = (bars.timestamp.iloc[-1] + pd.Timedelta(days=1)).tz_localize("UTC")
    csv = tmp_path / "TEST.csv"
    bars.to_csv(csv, index=False)
    training = tmp_path / "training.csv"
    bars.iloc[:90].to_csv(training, index=False)
    config = Config(
        symbol="TEST", sequence_length=5, epochs=1, hidden_size=4, device="cpu", as_of=now.date().isoformat()
    )
    trained = train_candidate(training, config, tmp_path / "models", tmp_path / "training")
    first = trained["registered_model"]
    trained = train_candidate(training, config, tmp_path / "models", tmp_path / "training", force=True)
    second = trained["registered_model"]
    policy = json.loads(Path("configs/release_policy.json").read_text())
    policy["symbol"] = "TEST"
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))

    # Production has no CLI clock override. Tests advance time to simulate matured future labels.
    class Clock:
        @staticmethod
        def now(tz=None):
            return now.to_pydatetime()

    monkeypatch.setattr(gate, "datetime", Clock)
    return client, first["name"], first["version"], second["version"], csv, policy_path, now


def perfect_predictions(bundle, frame, eligible):
    # Controlled oracle used only to exercise passing thresholds, never to claim model quality.
    return frame.target_return.iloc[eligible].to_numpy() + 0.000001


def test_bootstrap_packaging_and_confirmed_deployment(case, monkeypatch, tmp_path):
    client, name, first, second, csv, policy, now = case
    assert deployed_version(client, name) is None
    ordinary = gate.run_gate(client, name, second, csv, policy)
    assert ordinary["decision"] == "blocked"
    assert "no_deployed_model_use_explicit_bootstrap" in ordinary["reasons"]
    monkeypatch.setattr(gate, "predictions", perfect_predictions)
    result = gate.run_gate(client, name, second, csv, policy, bootstrap=True)
    assert result["decision"] == "passed", result
    assert result["eligible_n"] >= 120
    assert deployed_version(client, name) is None
    gate.verify_gate(client, result["gate_run_id"], policy, csv)
    # API smoke testing is covered separately; its wall clock cannot serve synthetic future data.
    monkeypatch.setattr(release, "predict_bundle", lambda *args: None)
    output = release.package_release(csv.parent, tmp_path / "release", client, result["gate_run_id"], policy)
    manifest = json.loads((output / "release.json").read_text())
    assert manifest["candidate_version"] == second
    assert len(list((output / "models").glob("*/weights.pt"))) == 1
    assert deployed_version(client, name) is None

    def failed_endpoint(*args):
        raise ValueError("wrong serving version")

    monkeypatch.setattr(deployment, "verify_serving", failed_endpoint)
    with pytest.raises(ValueError, match="wrong serving"):
        deployment.confirm_deployment(client, result["gate_run_id"], "https://example.test", policy)
    assert deployed_version(client, name) is None
    monkeypatch.setattr(deployment, "verify_serving", lambda *args: None)
    deployment.confirm_deployment(client, result["gate_run_id"], "https://example.test", policy)
    assert str(deployed_version(client, name)) == second
    assert gate.run_gate(client, name, second, csv, policy, bootstrap=True)["decision"] == "blocked"


def test_replacement_same_origins_and_no_refitting(case, monkeypatch):
    client, name, first, second, csv, policy, now = case
    client.set_registered_model_alias(name, "deployed", first)
    client.set_registered_model_tag(name, "deployment_endpoint", "https://example.test")
    monkeypatch.setattr(deployment, "verify_serving", lambda *args: None)
    seen = []
    original_predict = gate.predictions

    def controlled(bundle, frame, eligible):
        # Exercise the real feature/scaler/prediction path before controlling the score.
        assert np.isfinite(original_predict(bundle, frame, eligible)).all()
        seen.append(eligible.copy())
        return frame.target_return.iloc[eligible].to_numpy() + (0.000001 if str(bundle[4].version) == second else 0.01)

    from sklearn.preprocessing import StandardScaler

    monkeypatch.setattr(StandardScaler, "fit", lambda *args: pytest.fail("Evaluation must not fit a scaler"))
    monkeypatch.setattr(gate, "predictions", controlled)
    result = gate.run_gate(client, name, second, csv, policy)
    assert result["decision"] == "passed", result
    np.testing.assert_array_equal(seen[0], seen[1])
    assert result["metrics"]["candidate"]["rmse"] < result["metrics"]["deployed"]["rmse"]
    assert str(deployed_version(client, name)) == first
    with pytest.raises(ValueError, match="expired"):
        gate.verify_gate(client, result["gate_run_id"], policy, now=now + pd.Timedelta(days=2))
    client.set_registered_model_alias(name, "deployed", second)
    with pytest.raises(ValueError, match="Deployed version changed"):
        gate.verify_gate(client, result["gate_run_id"], policy)


def test_insufficient_stale_incompatible_and_missing_evidence(case, monkeypatch):
    client, name, first, second, csv, policy, now = case
    raw = pd.read_csv(csv)
    raw.iloc[:100].to_csv(csv, index=False)
    result = gate.run_gate(client, name, second, csv, policy, bootstrap=True)
    assert result["decision"] == "blocked"
    assert "insufficient_eligible_observations" in result["reasons"]
    assert "stale_evaluation_data" in result["reasons"]
    raw.to_csv(csv, index=False)
    from stock_forecasting import training

    monkeypatch.setattr(training, "objective", lambda *args: 0.0)
    refit_config = Config(
        symbol="TEST", sequence_length=5, epochs=1, hidden_size=4, device="cpu", as_of=now.date().isoformat()
    )
    refit = train_candidate(
        csv, refit_config, csv.parent / "refit-models", csv.parent / "refit-training", refit_production=True
    )
    retired = gate.run_gate(client, name, refit["registered_model"]["version"], csv, policy, bootstrap=True)
    assert retired["eligible_n"] == 0
    assert "insufficient_eligible_observations" in retired["reasons"]
    path, _, _, metadata, entry = registered_bundle(client, name, second)
    # Integrity failures cannot turn into approvals or bootstrap fallbacks.
    (path / "weights.pt").write_bytes(b"corrupt")
    client.log_artifact(entry.run_id, str(path / "weights.pt"), entry.tags["artifact_path"])
    result = gate.run_gate(client, name, second, csv, policy, bootstrap=True)
    assert result["decision"] == "blocked"
    assert "checksum" in str(result["reasons"]).lower()


def test_changed_data_policy_and_newer_gate_invalidate_approval(case, monkeypatch):
    client, name, first, second, csv, policy, now = case
    monkeypatch.setattr(gate, "predictions", perfect_predictions)
    result = gate.run_gate(client, name, second, csv, policy, bootstrap=True)
    assert result["decision"] == "passed"
    raw = csv.read_bytes()
    csv.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="data changed"):
        gate.verify_gate(client, result["gate_run_id"], policy, csv)
    csv.write_bytes(raw)
    original_policy = policy.read_text()
    policy.write_text(original_policy + "\n")
    with pytest.raises(ValueError, match="policy changed"):
        gate.verify_gate(client, result["gate_run_id"], policy)
    policy.write_text(original_policy)
    monkeypatch.setattr(gate, "predictions", lambda bundle, frame, eligible: np.full(len(eligible), np.nan))
    newer = gate.run_gate(client, name, second, csv, policy, bootstrap=True)
    assert newer["decision"] == "blocked"
    with pytest.raises(ValueError, match="newer evaluation"):
        gate.verify_gate(client, result["gate_run_id"], policy)
    from stock_forecasting.model_registry import read_gate

    forged = {**newer, "decision": "passed"}
    fake = csv.parent / "gate.json"
    fake.write_text(json.dumps(forged))
    client.log_artifact(newer["gate_run_id"], str(fake), "metadata")
    with pytest.raises(ValueError, match="modified"):
        read_gate(client, newer["gate_run_id"])


@pytest.mark.parametrize("rmse,mae,passed", [(0.99, 0.8, True), (0.99001, 0.8, False), (0.9, 0.80001, False)])
def test_threshold_boundaries(rmse, mae, passed):
    policy, _ = gate.policy_file("configs/release_policy.json")
    reasons = gate.threshold_failures(
        {"rmse": rmse, "mae": mae}, {"rmse": 1.1, "mae": 0.9}, {"rmse": 1.0, "mae": 0.8}, policy
    )
    assert (not reasons) == passed


def test_no_local_promotion_bypass(tmp_path):
    with pytest.raises(ValueError, match="release gate"):
        release.package_release(tmp_path, tmp_path / "release")


def test_unguarded_azure_path_is_closed(monkeypatch):
    from stock_forecasting import azure

    monkeypatch.setattr(azure, "ml_client", lambda: pytest.fail("Must block before contacting Azure"))
    with pytest.raises(ValueError, match="gate-aware"):
        azure.deploy("endpoint", "model", "image", "cpu")


@pytest.mark.parametrize("served_version", ["expected", "wrong"])
def test_serving_verification_checks_actual_response(monkeypatch, served_version):
    import httpx

    original_client = httpx.Client
    monkeypatch.setenv("FORECAST_API_KEY", "test-key")

    def response(request):
        assert request.headers["X-API-Key"] == "test-key"
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        assert request.url.path == "/v1/predict"
        assert "model_version" not in json.loads(request.content)
        return httpx.Response(200, json={"model_version": served_version})

    monkeypatch.setattr(
        deployment.httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(response), **kwargs)
    )
    metadata = {"model_version": "expected", "model": {"config": Config().to_dict()}}
    if served_version == "wrong":
        with pytest.raises(ValueError, match="Serving model differs"):
            deployment.verify_serving("https://example.test", metadata)
    else:
        deployment.verify_serving("https://example.test", metadata)
