"""Validate downloaded symbols, version datasets with DVC, train local LSTM candidates."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .config import Config
from .data import features, normalize, read_csv, sequences, split_masks, targets
from .training import train_candidate

PREPROCESSING_VERSION = "canonical-daily-v1"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def symbols_for(raw, requested=None):
    from utils.constants import SYMBOLS

    # Download batches may temporarily comment out symbols in constants.py.
    values = (
        requested
        if requested is not None
        else (
            set(SYMBOLS)
            | {p.stem for p in Path(raw).glob("*.csv")}
            | {p.name.removesuffix(".csv.dvc") for p in Path(raw).glob("*.csv.dvc")}
        )
    )
    values = sorted(set(values))
    for symbol in values:
        Config(symbol=symbol)  # Validate before constructing paths.
    return values


def archive_old_output(path, archive):
    """Preserve superseded outputs/pointers outside the active DVC workspace."""
    for item in (path, Path(str(path) + ".dvc")):
        if item.exists():
            destination = Path(archive) / f"{item.name}.{digest(item)[:16]}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.read_bytes() != item.read_bytes():
                    raise ValueError("Archive collision")
                item.unlink()
            else:
                item.replace(destination)


def preprocess_market(
    raw, processed, config, symbols=None, as_of=None, max_age_days=14, archive="artifacts/market/archive"
):
    raw, processed = Path(raw), Path(processed)
    if raw.resolve() == processed.resolve():
        raise ValueError("Raw and processed directories must differ")
    if max_age_days < 0:
        raise ValueError("max_age_days must be nonnegative")
    cutoff = pd.Timestamp(as_of or pd.Timestamp.now(tz="UTC"))
    cutoff = cutoff.tz_convert("UTC").tz_localize(None) if cutoff.tzinfo else cutoff
    if pd.isna(cutoff):
        raise ValueError("Invalid processing cutoff")
    cutoff = cutoff.normalize()
    processed.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "preprocessing_version": PREPROCESSING_VERSION,
        "as_of": str(cutoff.date()),
        "max_age_days": max_age_days,
        "minimum_samples": {"train": 252, "validation": 60, "test": 60},
        "symbols": {},
    }
    for symbol in symbols_for(raw, symbols):
        source, destination = raw / f"{symbol}.csv", processed / f"{symbol}.csv"
        entry = {"status": "invalid", "raw_path": source.as_posix(), "reason": ""}
        result["symbols"][symbol] = entry
        c = replace(config, symbol=symbol, as_of=result["as_of"])
        try:
            entry["raw_sha256"] = digest(source)
            frame, quality = normalize(read_csv(source), c)
            entry.update(quality)
            # Save canonical prices, not globally scaled features. Scaling belongs to each training fold.
            # Preserve unadjusted inputs so normalize() cannot apply adjustments twice during training.
            if c.adjustment == "adjusted":
                raise ValueError(
                    "Market preprocessing currently requires raw/unverified prices; adjusted inputs need an explicit contract"
                )
            canonical = frame[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
            temporary = destination.with_suffix(".csv.tmp")
            canonical.to_csv(temporary, index=False, date_format="%Y-%m-%d")
            temporary.replace(destination)
            entry.update(
                processed_sha256=digest(destination),
                processed_path=destination.as_posix(),
                preprocessing_version=PREPROCESSING_VERSION,
                status="ineligible",
            )
            engineered, columns = features(frame, c.feature_groups)
            samples = sequences(targets(engineered, c.horizon), columns, c.sequence_length, c.horizon)
            masks = split_masks(
                samples,
                int(len(frame) * c.train_fraction),
                int(len(frame) * (c.train_fraction + c.validation_fraction)),
            )
            counts = {name: int(mask.sum()) for name, mask in zip(("train", "validation", "test"), masks)}
            entry["sample_counts"] = counts
            entry["age_days"] = (cutoff - frame.timestamp.iloc[-1]).days
            if entry["age_days"] > max_age_days:
                entry["reason"] = f"Stale last observation ({entry['age_days']} days)"
            elif any(counts[k] < minimum for k, minimum in result["minimum_samples"].items()):
                entry["reason"] = "Insufficient samples for train/validation/holdout minimums"
            else:
                entry.update(status="eligible", reason="")
        except (OSError, ValueError, KeyError) as exc:
            entry["reason"] = str(exc)
            if "processed_sha256" not in entry:
                archive_old_output(destination, archive)
        print(f"{symbol}: {entry['status']} {entry['reason']}", flush=True)
    write_json(processed / "manifest.json", result)
    return result


def track_datasets(raw, processed, manifest):
    """Only local DVC caching/pointers: no Git commit, remote push or cloud calls."""
    paths = []
    for symbol, entry in manifest["symbols"].items():
        source = Path(raw) / f"{symbol}.csv"
        if source.exists():
            paths.append(str(source))
        if "processed_sha256" in entry:
            paths.append(str(Path(processed) / f"{symbol}.csv"))
    paths.append(str(Path(processed) / "manifest.json"))
    subprocess.run([sys.executable, "-m", "dvc", "add", *paths], check=True)


def report(path, manifest, results):
    counts = Counter(r["status"] for r in manifest["symbols"].values())
    lines = [
        "# Market-wide LSTM candidate training",
        "",
        f"Data cutoff (exclusive): {manifest['as_of']}",
        "",
        "Validation: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
        "Raw data is unchanged. No missing prices/sessions are filled. Invalid OHLC blocks a symbol. "
        "Stale/short histories are retained for research but excluded from current candidate training. "
        "Instrument type/listing status is not verified by this CSV validation.",
        "",
        "Training uses chronological 60/20/20 splits with horizon purging; scalers fit only training inputs. "
        "The final holdout is not scored or fitted. These are development candidates, not deployment approvals.",
        "",
        "| Symbol | Data status | Rows | Latest date | Training | Validation RMSE | Baseline RMSE | Reason |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for symbol, entry in manifest["symbols"].items():
        trained = results.get(symbol, {})
        metrics = trained.get("validation_metrics", {}).get("regression", {})
        baseline = trained.get("baseline_metrics", {}).get("regression", {})
        reason = str(entry.get("reason") or trained.get("error", "")).replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {symbol} | {entry['status']} | {entry.get('rows', '—')} | {entry.get('end', '—')} | "
            f"{trained.get('status', 'not trained')} | {metrics.get('rmse', '—')} | "
            f"{baseline.get('rmse', '—')} | {reason} |"
        )
    lines.extend(
        [
            "",
            "Full dataset lineage: `data/processed/manifest.json` (DVC tracked). "
            "Training records: `artifacts/market/training/`; model weights: `artifacts/models/`; "
            "MLflow contains each training run and registered candidate version.",
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--as-of", help="Exclusive completed-day cutoff, defaults to current UTC date")
    parser.add_argument("--max-age-days", type=int, default=14)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-dvc", action="store_true", help="Testing only: skip local DVC tracking")
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--report", default="reports/market_training/market_training_report.md")
    args = parser.parse_args(argv)
    config = Config.read(args.config)
    if config.family != "lstm" or config.task != "return":
        parser.error("Market candidates use LSTM return forecasting")
    manifest = preprocess_market(args.raw, args.processed, config, args.symbols, args.as_of, args.max_age_days)
    results = {}
    report(args.report, manifest, results)
    if not args.no_dvc:
        track_datasets(args.raw, args.processed, manifest)
    if args.prepare_only:
        return
    import torch

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    for symbol, entry in manifest["symbols"].items():
        if entry["status"] != "eligible":
            continue
        print(f"Training {symbol} on {config.device}...", flush=True)
        try:
            result = train_candidate(
                Path(args.processed) / f"{symbol}.csv",
                replace(config, symbol=symbol, as_of=manifest["as_of"]),
                output="artifacts/market/training",
                mlflow_uri=args.mlflow_uri,
            )
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        results[symbol] = result
        write_json("artifacts/market/results.json", results)
        report(args.report, manifest, results)
        print(f"{symbol}: {result['status']}", flush=True)
    if any(r["status"] == "failed" for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
