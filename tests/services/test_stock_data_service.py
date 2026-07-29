import os
import subprocess
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.services import stock_data_service
from src.services import (
    CompanyNotFoundError,
    DataUnavailableError,
    ModelUnavailableError,
)
from src import LSTMModel


ENV = {"WINDOW_SIZE": "1", "MODEL_NAME": "LSTM", "MODEL_LOCATION": "LOCAL"}


def _make_df():
    return pd.DataFrame({
        "date": ["2020-01-01"],
        "ouverture": [100],
        "haut": [100],
        "bas": [100],
        "cloture": [100],
        "volume": [0],
    })


def _trained_model(df):
    model = LSTMModel("AB")
    model.last_trained_date = max(pd.to_datetime(df["date"], format="%Y-%m-%d", dayfirst=True))
    return model


@patch("src.services.stock_data_service.os.listdir")
def test_list_companies_strips_dvc_suffix(mock_listdir):
    mock_listdir.return_value = ["AB.csv.dvc", "AL.csv.dvc", "notes.txt"]

    assert stock_data_service.list_companies() == ["AB", "AL"]


@patch("src.services.stock_data_service.os.path.exists")
def test_load_company_df_unknown_company(mock_exists):
    mock_exists.return_value = False

    with pytest.raises(CompanyNotFoundError):
        stock_data_service.load_company_df("UNKNOWN")


@patch("src.services.stock_data_service.pd.read_csv")
@patch("src.services.stock_data_service.subprocess.run")
@patch("src.services.stock_data_service.os.path.exists")
def test_load_company_df_pulls_missing_csv(mock_exists, mock_run, mock_read_csv):
    # .dvc file present, csv absent
    mock_exists.side_effect = [True, False]
    mock_read_csv.return_value = _make_df()

    df = stock_data_service.load_company_df("AB")

    mock_run.assert_called_once()
    assert df["date"].dtype.kind == "M"


@patch("src.services.stock_data_service.subprocess.run")
@patch("src.services.stock_data_service.os.path.exists")
def test_load_company_df_reports_failed_pull(mock_exists, mock_run):
    mock_exists.side_effect = [True, False]
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="dvc pull", output="", stderr="no remote"
    )

    with pytest.raises(DataUnavailableError, match="no remote"):
        stock_data_service.load_company_df("AB")


@patch.dict(os.environ, ENV)
@patch("src.services.stock_data_service.get_scaler")
@patch("src.services.stock_data_service.get_model")
@patch("src.services.stock_data_service.pd.read_csv")
@patch("src.services.stock_data_service.os.path.exists")
def test_get_prediction_requires_a_model(mock_exists, mock_read_csv, mock_get_model, mock_get_scaler):
    mock_exists.return_value = True
    mock_read_csv.return_value = _make_df()
    mock_get_model.return_value = None
    mock_get_scaler.return_value = MinMaxScaler()

    with pytest.raises(ModelUnavailableError):
        stock_data_service.get_prediction("AB")


@patch.dict(os.environ, ENV)
@patch("src.services.stock_data_service.get_scaler")
@patch("src.services.stock_data_service.get_model")
@patch("src.services.stock_data_service.pd.read_csv")
@patch("src.services.stock_data_service.os.path.exists")
def test_get_prediction_rejects_a_stale_model(mock_exists, mock_read_csv, mock_get_model, mock_get_scaler):
    df = _make_df()
    mock_exists.return_value = True
    mock_read_csv.return_value = df
    model = LSTMModel("AB")
    model.last_trained_date = pd.Timestamp("2019-01-01")
    mock_get_model.return_value = model
    mock_get_scaler.return_value = MinMaxScaler()

    with pytest.raises(ModelUnavailableError):
        stock_data_service.get_prediction("AB")


@patch.dict(os.environ, ENV)
@patch("src.services.stock_data_service.predict")
@patch("src.services.stock_data_service.get_scaler")
@patch("src.services.stock_data_service.get_model")
@patch("src.services.stock_data_service.pd.read_csv")
@patch("src.services.stock_data_service.os.path.exists")
def test_get_prediction_returns_dated_forecast(
    mock_exists, mock_read_csv, mock_get_model, mock_get_scaler, mock_predict
):
    df = _make_df()
    mock_exists.return_value = True
    mock_read_csv.return_value = df
    mock_get_model.return_value = _trained_model(df)
    mock_get_scaler.return_value = MinMaxScaler()
    mock_predict.return_value = [110, 120]

    forecast = stock_data_service.get_prediction("AB", horizon=2)

    assert forecast["company"] == "AB"
    assert forecast["last_close"] == 100
    assert forecast["values"] == [110, 120]
    assert forecast["dates"] == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]


@patch.dict(os.environ, ENV)
@patch("src.services.stock_data_service.predict")
@patch("src.services.stock_data_service.get_scaler")
@patch("src.services.stock_data_service.get_model")
@patch("src.services.stock_data_service.pd.read_csv")
@patch("src.services.stock_data_service.os.path.exists")
def test_get_stock_with_forecast_appends_predicted_rows(
    mock_exists, mock_read_csv, mock_get_model, mock_get_scaler, mock_predict
):
    df = _make_df()
    mock_exists.return_value = True
    mock_read_csv.return_value = df
    mock_get_model.return_value = _trained_model(df)
    mock_get_scaler.return_value = MinMaxScaler()
    mock_predict.return_value = [110]

    res = stock_data_service.get_stock_with_forecast("AB")

    assert res["columns"] == ["date", "ouverture", "haut", "bas", "cloture", "volume"]
    assert len(res["data"]) == 2
    assert res["data"][1]["cloture"] == 110
    assert res["data"][1]["date"] == pd.Timestamp("2020-01-02")
