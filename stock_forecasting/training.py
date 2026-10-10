"""Routine candidate training; development validation gates local promotion.

The final research holdout is neither scored nor used to train this candidate.
Optional production refit retires the historical holdout after freezing selection.
"""

from hashlib import sha256
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .artifacts import save_bundle, load_bundle
from .benchmark import prepare, split_metadata, objective
from .data import fit_scaler, split_masks, transform, FEATURE_VERSION
from .models import create_forecaster
from .evaluation import evaluate
from .registry import identity, promote


def train_candidate(
    csv, config, root="artifacts/models", output="reports/training", force=False, refit_production=False
):
    Path(root).mkdir(parents=True, exist_ok=True)
    try:
        with (Path(root) / "active.json").open("x", encoding="utf-8") as stream:
            json.dump({"active": {}, "history": []}, stream)
    except FileExistsError:
        pass
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame, columns, samples, manifest = prepare(csv, config, output / "storage")
    implementation = b"".join(
        p.read_bytes()
        for folder in (Path(__file__).parent, Path(__file__).parents[1] / "src/prediction_model/models")
        for p in sorted(folder.glob("*.py"))
    )
    fingerprint = sha256(
        manifest["dataset_version"].encode()
        + json.dumps(config.to_dict(), sort_keys=True).encode()
        + FEATURE_VERSION.encode()
        + implementation
        + str(refit_production).encode()
    ).hexdigest()
    record = output / f"{identity(config)}-{fingerprint}.json"
    if record.exists() and not force:
        result = json.loads(record.read_text(encoding="utf-8"))
        load_bundle(Path(root) / result["model_version"])
        return {**result, "status": "unchanged"}
    train, val, test = split_masks(
        samples,
        int(len(frame) * config.train_fraction),
        int(len(frame) * (config.train_fraction + config.validation_fraction)),
    )
    scaler, rows = fit_scaler(frame, columns, samples, train, config.sequence_length)
    x = transform(samples.x, scaler)
    model = create_forecaster(len(columns), config).fit(x[train], samples.y[train], x[val], samples.y[val])
    prediction = model.predict(x[val])
    baseline = {
        "return": np.full(val.sum(), samples.y[train].mean()) if config.task != "direction" else None,
        "probability": np.full(val.sum(), np.mean(samples.y[train] > 0)) if config.task == "direction" else None,
    }
    score, baseline_score = objective(samples.y[val], prediction), objective(samples.y[val], baseline)
    accepted = bool(np.isfinite(score) and score <= baseline_score)
    bundle = save_bundle(
        model, scaler, columns, manifest, split_metadata(frame, samples, (train, val, test), rows), root
    )
    load_bundle(bundle)
    result = {
        "status": "promoted" if accepted else "rejected",
        "model_version": bundle.name,
        "fingerprint": fingerprint,
        "identity": identity(config),
        "validation_objective": score,
        "validation_metrics": evaluate(samples.y[val], prediction, samples.y[train]),
        "baseline_objective": baseline_score,
        "policy": "validation MSE or Brier <= train-only mean baseline",
        "fit_policy": "development_only_no_production_refit",
        "holdout_used": False,
    }
    result["candidate_version"] = bundle.name
    if accepted and refit_production:
        # Selection is frozen before refitting. The old holdout is retired as research data.
        epochs = min(model.history, key=lambda row: row["validation_loss"])["epoch"]
        production_config = replace(config, epochs=epochs)
        all_matured = np.ones(len(samples.y), dtype=bool)
        production_scaler, production_rows = fit_scaler(frame, columns, samples, all_matured, config.sequence_length)
        production_x = transform(samples.x, production_scaler)
        production = create_forecaster(len(columns), production_config).fit_fixed(production_x, samples.y)
        split = split_metadata(frame, samples, (train, val, test), rows)
        split.update(
            train_last_label=str(frame.timestamp.iloc[samples.ends[-1]]),
            scaler_last_input=str(frame.timestamp.iloc[production_rows[-1]]),
            production_refit_n=len(samples.y),
            production_refit=True,
            research_holdout_retired=True,
            selected_epochs=epochs,
        )
        bundle = save_bundle(production, production_scaler, columns, manifest, split, root)
        result.update(
            model_version=bundle.name,
            fit_policy="all_matured_labels_fixed_selected_epochs",
            holdout_used=True,
            holdout_scored=False,
            selected_epochs=epochs,
        )
    if accepted:
        promote(root, bundle.name, identity(config))
    record.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
