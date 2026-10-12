import json
import pytest
from scripts import train_model
from stock_forecasting.config import Config


def config_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(Config(symbol="TEST", device="cpu").to_dict()))
    return path


def test_training_script_records_each_symbol_and_continues_failures(tmp_path, monkeypatch):
    calls = []

    def candidate(data, config, *args):
        calls.append((data.name, config.symbol))
        if config.symbol == "BAD":
            raise ValueError("Insufficient history")
        return {"symbol": config.symbol, "status": "rejected"}

    monkeypatch.setattr(train_model, "train_candidate", candidate)
    output = tmp_path / "reports"
    with pytest.raises(SystemExit) as error:
        train_model.main(["--config", str(config_file(tmp_path)), "--symbols", "BAD", "TEST", "--output", str(output)])
    assert error.value.code == 1
    assert calls == [("BAD.csv", "BAD"), ("TEST.csv", "TEST")]
    report = json.loads((output / "run.json").read_text())
    assert [entry["status"] for entry in report] == ["failed", "rejected"]


def test_azure_submission_requires_explicit_flag(tmp_path, monkeypatch):
    import stock_forecasting.azure as azure

    def unexpected(*args):
        pytest.fail("Azure must not be contacted")

    monkeypatch.setattr(azure, "submit_training", unexpected)
    with pytest.raises(SystemExit) as error:
        train_model.main(
            [
                "--config",
                str(config_file(tmp_path)),
                "--backend",
                "azure",
                "--data",
                "azureml:data:1",
                "--compute",
                "cpu",
            ]
        )
    assert error.value.code == 2


def test_single_file_rejects_multiple_symbols(tmp_path):
    with pytest.raises(SystemExit) as error:
        train_model.main(["--config", str(config_file(tmp_path)), "--data", "some.csv", "--symbols", "A", "B"])
    assert error.value.code == 2
