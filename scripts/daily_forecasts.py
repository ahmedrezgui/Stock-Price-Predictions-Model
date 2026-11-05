"""Publish a manually downloaded, validated daily snapshot and run remote forecasts.

Defaults to preflight only. No training, scraping, or local prediction fallback.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

from stock_assistant.forecast_client import ForecastServiceError, get_models, get_prediction
from stock_forecasting.config import Config
from stock_forecasting.data import normalize, read_csv


def preflight(source, catalog, expected_through=None, symbols=None):
    models = [m for m in catalog if m.get("model") == "lstm" and m.get("task") == "return" and m.get("horizon") == 1]
    if symbols:
        missing = set(symbols) - {m["symbol"] for m in models}
        if missing:
            raise ValueError(f"No deployed one-observation model for: {', '.join(sorted(missing))}")
        models = [m for m in models if m["symbol"] in symbols]
    if not models:
        raise ValueError("No deployed models available")
    rows = []
    for model in models:
        symbol = model["symbol"]
        Config(symbol=symbol)  # Validate before constructing filesystem paths.
        path = Path(source) / f"{symbol}.csv"
        frame, metadata = normalize(read_csv(path), Config(symbol=symbol))
        latest = frame.timestamp.iloc[-1]
        if expected_through and latest < pd.Timestamp(expected_through):
            raise ValueError(f"{symbol}: latest observation {latest.date()} is before required {expected_through}")
        if (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - latest).days > 14:
            raise ValueError(f"{symbol}: serving data is older than 14 calendar days")
        rows.append({"symbol": symbol, "model_version": model["model_version"], "as_of": str(latest.date()),
                     "rows": metadata["rows"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def fresh_prediction(row, predict=get_prediction, sleep=time.sleep):
    # Serving data is cached for 60 seconds. Never silently record the old result
    # as the newly requested forecast while that cache is expiring.
    for attempt in range(5):
        result = predict(row["symbol"], model_version=row["model_version"])
        if result["as_of"] == row["as_of"]:
            return result
        if result["as_of"] > row["as_of"]:
            raise ValueError("Serving data advanced during this run; rerun preflight")
        if attempt < 4:
            sleep(20)
    raise ValueError(f"{row['symbol']}: API has not refreshed to {row['as_of']}")


def upload_chart_snapshot(path, symbol, account, share, resource_group):
    """Use the existing Azure CLI login; never expose the account key in argv/logs."""
    az = shutil.which("az") or shutil.which("az.cmd")
    if not az:
        raise RuntimeError("Azure CLI is required for the dashboard file-share upload")
    keys = subprocess.run([az, "storage", "account", "keys", "list", "-g", resource_group, "-n", account, "-o", "json"], capture_output=True, text=True)
    if keys.returncode:
        raise RuntimeError("Cannot obtain dashboard storage credentials using the Azure CLI login")
    env = {**os.environ, "AZURE_STORAGE_ACCOUNT": account, "AZURE_STORAGE_KEY": json.loads(keys.stdout)[0]["value"]}
    # The Blob publisher's account URL must not override the different Files
    # account endpoint inherited by Azure CLI.
    for name in ("AZURE_STORAGE_ACCOUNT_URL", "AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_SAS_TOKEN"):
        env.pop(name, None)
    uploaded = subprocess.run([az, "storage", "file", "upload", "--account-name", account, "--share-name", share, "--path", f"data/{symbol}.csv", "--source", str(path), "--no-progress", "--only-show-errors", "-o", "none"], env=env, capture_output=True, text=True)
    if uploaded.returncode:
        detail = uploaded.stderr.replace(env["AZURE_STORAGE_KEY"], "[redacted]")[:1000]
        raise RuntimeError(f"Dashboard snapshot upload failed for {symbol}: {detail}")


def run(source, rows, publish, upload, predict=fresh_prediction):
    results = []
    # Freeze all validated bytes before any publication; a concurrent manual
    # download must not change files between validation and upload.
    with tempfile.TemporaryDirectory(prefix="stockwise-daily-") as temporary:
        frozen = Path(temporary)
        for row in rows:
            data = (Path(source) / f"{row['symbol']}.csv").read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Source changed after validation; rerun preflight")
            (frozen / f"{row['symbol']}.csv").write_bytes(data)
        for row in rows:
            item = {**row, "status": "pending"}
            try:
                path = frozen / f"{row['symbol']}.csv"
                publish(path, row["symbol"])
                item["serving_published"] = True
                upload(path, row["symbol"])
                item["dashboard_updated"] = True
                item["forecast"] = predict(row)
                item["status"] = "success"
            except Exception as exc:
                item["status"] = "failed"
                # Avoid dumping service exception bodies that may contain URLs/tokens.
                item["error"] = type(exc).__name__
                if isinstance(exc, RuntimeError) and str(exc).startswith(("Dashboard snapshot", "Cannot obtain dashboard", "Azure CLI is required")):
                    item["detail"] = str(exc)
                item["action"] = "Check Azure access, source freshness and service health, then rerun. Earlier successful symbols remain published."
            results.append(item)
    return results


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw")
    parser.add_argument("--download-from", help="Optionally run the existing interactive ilboursa downloader first (dd-mm-yyyy)")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--expected-through", help="Last completed observation date, YYYY-MM-DD; required with --apply")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", default="reports/daily/latest.json")
    parser.add_argument("--forecast-storage-url", default="https://ststockhistoricdata.blob.core.windows.net")
    parser.add_argument("--forecast-container", default="forecast-serving")
    parser.add_argument("--dashboard-storage", default="stockwiseapprezgu261014")
    parser.add_argument("--dashboard-share", default="stockwise-app")
    parser.add_argument("--resource-group", default="rg-stock-forecast-student")
    args = parser.parse_args()
    if args.apply and not args.expected_through:
        parser.error("--apply requires --expected-through so missing new observations cannot pass silently")
    if args.expected_through:
        expected = datetime.strptime(args.expected_through, "%Y-%m-%d").date()
        if expected >= datetime.now(timezone.utc).date():
            parser.error("Use a completed observation date before today; current-day bars are excluded")
    catalog = get_models()
    if args.download_from:
        selected = args.symbols or sorted({m["symbol"] for m in catalog if m.get("model") == "lstm" and m.get("task") == "return" and m.get("horizon") == 1})
        if not selected:
            parser.error("No deployed models to refresh")
        subprocess.run([sys.executable, "-m", "scripts.download_data", "--date", args.download_from,
                        "--symbols", *selected, "--output-dir", args.source], check=True)
    rows = preflight(args.source, catalog, args.expected_through, args.symbols)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "mode": "apply" if args.apply else "preflight", "stocks": rows}
    if args.apply:
        from stock_forecasting.cloud_storage import publish_data
        os.environ["AZURE_STORAGE_ACCOUNT_URL"] = args.forecast_storage_url
        report["stocks"] = run(args.source, rows,
            lambda path, symbol: publish_data(path, symbol, args.forecast_container),
            lambda path, symbol: upload_chart_snapshot(path, symbol, args.dashboard_storage, args.dashboard_share, args.resource_group))
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    failed = sum(row.get("status") == "failed" for row in report["stocks"])
    print(f"{report['mode']}: {len(rows)} stocks; {failed} failures. Report: {target}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ForecastServiceError) as exc:
        raise SystemExit(f"Daily refresh stopped: {exc}") from None
