"""Fail-closed, prospective comparison of exact registered model versions."""

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import features, normalize, read_csv, sequences, targets, transform
from .model_registry import bundle_digest, deployed_version, registered_bundle, read_gate
from .storage import snapshot_identity
from .tracking import tracking_run


def utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Missing evidence timestamp")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def policy_file(path):
    raw = Path(path).read_bytes()
    policy = json.loads(raw)
    if policy["schema_version"] != 1 or policy["task"] != "return" or policy["horizon"] != 1:
        raise ValueError("This release policy supports one-bar return forecasts only")
    if policy["min_observations"] < 120:
        raise ValueError("At least 120 paired observations are required")
    if not 0 < policy["min_rmse_improvement"] < 1 or policy["max_mae_degradation"] != 0:
        raise ValueError("Require positive RMSE improvement and no MAE degradation")
    if policy["max_data_age_days"] <= 0 or not 0 < policy["gate_valid_hours"] <= 24:
        raise ValueError("Invalid evidence freshness/expiry policy")
    utc(policy["frozen_at"])
    return policy, sha256(raw).hexdigest()


def errors(truth, prediction):
    if not len(truth) or not np.isfinite(truth).all():
        raise ValueError("Finite, nonempty outcomes are required")
    prediction = np.asarray(prediction)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("Incomplete, misaligned or nonfinite predictions")
    residual = prediction - truth
    return {"rmse": float(np.sqrt(np.mean(residual**2))), "mae": float(np.mean(abs(residual)))}


def threshold_failures(candidate, baseline, incumbent, policy):
    reasons = []
    if candidate["rmse"] >= baseline["rmse"]:
        reasons.append("candidate_does_not_beat_mean_baseline")
    comparator = incumbent if incumbent is not None else baseline
    if comparator["rmse"] <= 0 or candidate["rmse"] > comparator["rmse"] * (1 - policy["min_rmse_improvement"]):
        reasons.append("insufficient_rmse_improvement")
    if candidate["mae"] > comparator["mae"] * (1 + policy["max_mae_degradation"]):
        reasons.append("mae_degradation")
    return reasons


def known_through(bundle):
    _, _, _, metadata, entry = bundle
    split = metadata["split"]
    # Fail on missing lineage, including older bundles without creation timestamps.
    return max(
        utc(metadata["created_at"]),
        utc(split["train_last_label"]),
        utc(split["validation_last_label"]),
        pd.Timestamp(entry.creation_timestamp, unit="ms", tz="UTC"),
    )


def predictions(bundle, frame, eligible):
    _, model, scaler, metadata, _ = bundle
    engineered, columns = features(frame, model.config.feature_groups)
    if columns != metadata["feature_columns"]:
        raise ValueError("Feature order no longer matches the frozen model")
    engineered = targets(engineered, model.config.horizon)
    samples = sequences(engineered, columns, model.config.sequence_length, model.config.horizon)
    indexes = pd.Index(samples.origins).get_indexer(eligible)
    if (indexes < 0).any():
        raise ValueError("A model cannot predict every eligible origin")
    # Only transform with saved scalers; neither model nor preprocessing is refitted.
    return model.predict(transform(samples.x[indexes], scaler))["return"]


def evaluate_versions(client, name, version, csv, policy, now, bootstrap=False):
    current = deployed_version(client, name)
    report = {
        "registry_name": name,
        "candidate_version": str(version),
        "deployed_version": str(current) if current else None,
        "mode": "bootstrap" if bootstrap else "replacement",
        "reasons": [],
    }
    if bootstrap and current:
        report["reasons"].append("bootstrap_forbidden_when_deployed_model_exists")
    if not bootstrap and not current:
        report["reasons"].append("no_deployed_model_use_explicit_bootstrap")
    if str(version) == str(current):
        report["reasons"].append("candidate_is_already_deployed")
    candidate = registered_bundle(client, name, version)
    incumbent = registered_bundle(client, name, current) if current else None
    if incumbent:
        from .deployment import verify_serving

        endpoint = client.get_registered_model(name).tags.get("deployment_endpoint")
        if not endpoint:
            raise ValueError("Deployed alias has no verified serving endpoint")
        verify_serving(endpoint, incumbent[3])
    bundles = [candidate] + ([incumbent] if incumbent else [])
    config = candidate[1].config
    for bundle in bundles:
        other = bundle[1].config
        if (other.symbol, other.horizon, other.task) != (policy["symbol"], policy["horizon"], policy["task"]):
            raise ValueError("Model and policy identities differ")
        if (other.adjustment, other.calendar, other.session_file) != (
            config.adjustment,
            config.calendar,
            config.session_file,
        ):
            raise ValueError("Models have incompatible price/calendar definitions")
    cutoff = max(utc(policy["frozen_at"]), *(known_through(bundle) for bundle in bundles))
    # Origins must be on a later date than any training, selection, registration or policy freeze.
    evaluation_config = replace(config, start=None, end=None, as_of=now.date().isoformat())
    frame, quality = normalize(read_csv(csv), evaluation_config)
    quality["source_file_sha256"] = sha256(Path(csv).read_bytes()).hexdigest()
    manifest, _ = snapshot_identity(frame, quality)
    frame = targets(frame, config.horizon)
    eligible = np.flatnonzero((frame.timestamp > cutoff.tz_localize(None).normalize()) & frame.target_return.notna())
    report.update(
        dataset=manifest,
        eligible_n=len(eligible),
        excluded_through=cutoff.isoformat(),
        candidate_bundle_sha256=bundle_digest(candidate[0]),
        deployed_bundle_sha256=bundle_digest(incumbent[0]) if incumbent else None,
    )
    if len(eligible) < policy["min_observations"]:
        report["reasons"].append("insufficient_eligible_observations")
    if now.date() - frame.timestamp.iloc[-1].date() > timedelta(days=policy["max_data_age_days"]):
        report["reasons"].append("stale_evaluation_data")
    if report["reasons"]:
        return report
    truth = frame.target_return.iloc[eligible].to_numpy()
    candidate_pred = predictions(candidate, frame, eligible)
    deployed_pred = predictions(incumbent, frame, eligible) if incumbent else None
    # Frozen training-only mean, not the mean of evaluation outcomes.
    baseline = np.full(len(truth), candidate[3]["model"]["target_mean"])
    scores = {"candidate": errors(truth, candidate_pred), "baseline": errors(truth, baseline)}
    if incumbent:
        scores["deployed"] = errors(truth, deployed_pred)
    report.update(
        metrics=scores,
        first_origin=str(frame.timestamp.iloc[eligible[0]]),
        last_origin=str(frame.timestamp.iloc[eligible[-1]]),
        last_label=str(frame.target_timestamp.iloc[eligible[-1]]),
    )
    report["reasons"] += threshold_failures(scores["candidate"], scores["baseline"], scores.get("deployed"), policy)
    report["paired_predictions"] = {
        "origin": frame.timestamp.iloc[eligible].astype(str).tolist(),
        "label_date": frame.target_timestamp.iloc[eligible].astype(str).tolist(),
        "actual": truth.tolist(),
        "candidate": np.asarray(candidate_pred).tolist(),
        "deployed": np.asarray(deployed_pred).tolist() if incumbent else None,
    }
    return report


def run_gate(client, name, version, csv, policy_path="configs/release_policy.json", bootstrap=False, now=None):
    policy, digest = policy_file(policy_path)
    now = utc(now or datetime.now(timezone.utc))
    config = Config(symbol=policy["symbol"], horizon=policy["horizon"], task=policy["task"])
    with tracking_run(config, "release_gate", client.tracking_uri) as tracking:
        report = {"registry_name": name, "candidate_version": str(version), "reasons": []}
        try:
            report = evaluate_versions(client, name, version, csv, policy, now, bootstrap)
        except Exception as exc:
            # Operational errors and missing lineage are evidence failures, not passes.
            report["reasons"].append(f"evaluation_error: {type(exc).__name__}: {exc}")
        paired = report.pop("paired_predictions", None)
        if paired:
            tracking.document(paired, "paired_predictions.json")
        report.update(
            decision="passed" if not report["reasons"] else "blocked",
            policy=policy,
            policy_sha256=digest,
            evaluated_at=now.isoformat(),
            expires_at=(now + timedelta(hours=policy["gate_valid_hours"])).isoformat(),
            gate_run_id=tracking.run_id,
        )
        tracking.params({"registry_name": name, "candidate_registry_version": str(version), "policy_sha256": digest})
        tracking.metrics(report.get("metrics", {}))
        tracking.metrics({"eligible_n": report.get("eligible_n", 0)})
        tracking.tags({"gate_decision": report["decision"], "gate_mode": "bootstrap" if bootstrap else "replacement"})
        tracking.params({"gate_evidence_sha256": sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()})
        tracking.document(report, "gate.json")
        # Only registry metadata changes. A pass does not set the deployed alias.
        client.set_model_version_tag(name, str(version), "release_status", report["decision"])
        client.set_model_version_tag(name, str(version), "latest_gate_run_id", tracking.run_id)
    return report


def verify_gate(client, gate_run_id, policy_path, csv=None, now=None):
    """Fetch authoritative evidence; never trust an editable local 'passed' flag."""
    report = read_gate(client, gate_run_id)
    _, digest = policy_file(policy_path)
    now = utc(now or datetime.now(timezone.utc))
    if report["decision"] != "passed" or report["policy_sha256"] != digest:
        raise ValueError("Release gate is blocked or policy changed")
    if not utc(report["evaluated_at"]) <= now <= utc(report["expires_at"]):
        raise ValueError("Release gate evidence expired or is from the future")
    name, candidate_version = report["registry_name"], report["candidate_version"]
    current = deployed_version(client, name)
    if (str(current) if current else None) != report["deployed_version"]:
        raise ValueError("Deployed version changed; evaluate again")
    candidate = registered_bundle(client, name, candidate_version)
    if candidate[4].tags.get("latest_gate_run_id") != gate_run_id:
        raise ValueError("A newer evaluation supersedes this gate")
    if bundle_digest(candidate[0]) != report["candidate_bundle_sha256"]:
        raise ValueError("Candidate bundle changed since evaluation")
    if current:
        incumbent = registered_bundle(client, name, current)
        if bundle_digest(incumbent[0]) != report["deployed_bundle_sha256"]:
            raise ValueError("Deployed bundle changed since evaluation")
    if csv is not None and sha256(Path(csv).read_bytes()).hexdigest() != report["dataset"]["source_file_sha256"]:
        raise ValueError("Evaluation data changed since the gate")
    return report, candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlflow-uri", required=True)
    parser.add_argument("--name", default="stock-BIAT-return-h1")
    parser.add_argument("--version", required=True, help="Exact registry version, not a mutable alias")
    parser.add_argument("--data", required=True)
    parser.add_argument("--policy", default="configs/release_policy.json")
    parser.add_argument("--bootstrap", action="store_true", help="First deployment only; still requires fresh evidence")
    args = parser.parse_args()
    from mlflow import MlflowClient

    result = run_gate(
        MlflowClient(tracking_uri=args.mlflow_uri), args.name, args.version, args.data, args.policy, args.bootstrap
    )
    print(json.dumps(result, indent=2))
    if result["decision"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
