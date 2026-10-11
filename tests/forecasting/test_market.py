from dataclasses import replace
from hashlib import sha256
import json

import pandas as pd
import pytest

from stock_forecasting.benchmark import prepare
from stock_forecasting.config import Config
from stock_forecasting.market import preprocess_market, report, symbols_for, track_datasets


@pytest.fixture
def market_bars(bars):
    frame = pd.concat([bars] * 3, ignore_index=True)
    frame["timestamp"] = pd.bdate_range("2023-01-02", periods=len(frame))
    return frame


def test_preprocessing_lineage_and_training_only_scaling(tmp_path, market_bars):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    source = raw / "TEST.csv"
    # Verify French names, decimal commas, duplicates, ordering and no raw mutations.
    original = market_bars.iloc[::-1].rename(
        columns={
            "symbol": "symbole",
            "timestamp": "date",
            "open": "ouverture",
            "high": "haut",
            "low": "bas",
            "close": "cloture",
        }
    )
    pd.concat([original, original.iloc[:1]]).to_csv(source, index=False, sep=";", decimal=",")
    before = source.read_bytes()
    cutoff = market_bars.timestamp.iloc[-1] + pd.Timedelta(days=1)
    config = Config(symbol="TEST", device="cpu")
    result = preprocess_market(raw, processed, config, ["TEST"], as_of=cutoff)
    entry = result["symbols"]["TEST"]
    assert entry["status"] == "eligible"
    assert source.read_bytes() == before
    canonical = pd.read_csv(processed / "TEST.csv")
    assert len(canonical) == len(market_bars)
    assert canonical.timestamp.is_monotonic_increasing
    assert list(canonical.columns) == ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    assert canonical.close.iloc[-1] == pytest.approx(market_bars.close.iloc[-1])
    _, _, _, manifest = prepare(
        processed / "TEST.csv", replace(config, as_of=str(cutoff.date())), tmp_path / "snapshots"
    )
    assert manifest["raw_source_sha256"] == sha256(before).hexdigest()
    assert manifest["source_file_sha256"] == entry["processed_sha256"]
    with (processed / "TEST.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="differs"):
        prepare(processed / "TEST.csv", config, tmp_path / "snapshots")


def test_invalid_stale_short_and_missing_are_reported(tmp_path, market_bars):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    bad = market_bars.assign(symbol="BAD").copy()
    bad.loc[0, "high"] = 0.1
    bad.to_csv(raw / "BAD.csv", index=False)
    market_bars.assign(symbol="STALE").iloc[:400].to_csv(raw / "STALE.csv", index=False)
    market_bars.assign(symbol="SHORT").iloc[-100:].to_csv(raw / "SHORT.csv", index=False)
    (processed / "BAD.csv").write_text("old data")
    (processed / "BAD.csv.dvc").write_text("old pointer")
    archive = tmp_path / "archive"
    result = preprocess_market(
        raw,
        processed,
        Config(),
        ["BAD", "STALE", "SHORT", "MISSING"],
        as_of=market_bars.timestamp.iloc[-1] + pd.Timedelta(days=1),
        archive=archive,
    )
    entries = result["symbols"]
    assert entries["BAD"]["status"] == entries["MISSING"]["status"] == "invalid"
    assert "Stale" in entries["STALE"]["reason"]
    assert entries["SHORT"]["status"] == "ineligible"
    assert not (processed / "BAD.csv").exists()
    assert len(list(archive.iterdir())) == 2
    assert (processed / "STALE.csv").exists()  # Valid research data remains available.
    report(tmp_path / "report.md", result, {})
    assert "OHLC range inconsistency" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_universe_includes_downloads_and_existing_dvc_pointers(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.constants.SYMBOLS", ["CONFIGURED"])
    (tmp_path / "DOWNLOADED.csv").touch()
    (tmp_path / "MISSING.csv.dvc").touch()
    assert symbols_for(tmp_path) == ["CONFIGURED", "DOWNLOADED", "MISSING"]
    with pytest.raises(ValueError):
        symbols_for(tmp_path, ["../escape"])


def test_dvc_tracks_raw_valid_processed_and_manifest_only(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    (raw / "BAD.csv").touch()
    (raw / "GOOD.csv").touch()
    calls = []
    monkeypatch.setattr("stock_forecasting.market.subprocess.run", lambda args, **kwargs: calls.append(args))
    track_datasets(raw, processed, {"symbols": {"BAD": {}, "GOOD": {"processed_sha256": "hash"}}})
    assert calls[0][1:4] == ["-m", "dvc", "add"]
    assert str(processed / "GOOD.csv") in calls[0]
    assert str(processed / "BAD.csv") not in calls[0]
    assert str(processed / "manifest.json") in calls[0]


def test_market_continues_training_failures_and_writes_progress(tmp_path, market_bars, monkeypatch):
    from stock_forecasting.market import main

    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    for symbol in ("BAD", "GOOD"):
        market_bars.assign(symbol=symbol).to_csv(raw / f"{symbol}.csv", index=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(Config(device="cpu").to_dict()))
    calls = []

    def train(csv, c, **kwargs):
        calls.append(c.symbol)
        assert csv.parent == tmp_path / "processed"
        if c.symbol == "BAD":
            raise ValueError("training failure")
        return {"status": "candidate"}

    monkeypatch.setattr("stock_forecasting.market.train_candidate", train)
    cutoff = str((market_bars.timestamp.iloc[-1] + pd.Timedelta(days=1)).date())
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--raw",
                str(raw),
                "--processed",
                str(tmp_path / "processed"),
                "--config",
                str(config),
                "--symbols",
                "BAD",
                "GOOD",
                "--as-of",
                cutoff,
                "--no-dvc",
            ]
        )
    assert exc.value.code == 1 and calls == ["BAD", "GOOD"]
    assert json.loads((tmp_path / "artifacts/market/results.json").read_text())["GOOD"]["status"] == "candidate"
