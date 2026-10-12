import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import pytest
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# we need to change directory to import modules
import sys,os
from pathlib import Path
sys.path.insert(0, str(Path(os.getcwd()) / '..' / '..'))
from src import IModel
from utils import split_dataset, get_features_target_from_dataset, plot_stock_graph, train, evaluate, plot_evaluation_result, train_model, predict
from enums import ValidationMetricEnum

def test_split_dataset():
    data = pd.DataFrame({'a': range(10)})
    train_df, test_df = split_dataset(data, test_size=0.3)
    assert len(train_df) == 7
    assert len(test_df) == 3


@pytest.mark.parametrize('one_dimensional', [False, True])
def test_train_and_evaluate_numpy_price_series(one_dimensional):
    from src import ARIMAModel

    prices = np.arange(100, dtype=float)
    if not one_dimensional:
        prices = prices.reshape(-1, 1)
    arima = Mock(spec=ARIMAModel)
    recurrent = Mock(spec=IModel)
    models = {'ARIMA': (arima, prices, prices), 'GRU': (recurrent, prices, prices)}
    scalers = train(models, window_size=60)
    arima_x, arima_y, _ = arima.train.call_args.args
    assert arima_x.shape == arima_y.shape == (100,)
    x, y, _ = recurrent.train.call_args.args
    assert x.shape == (40, 60, 1)
    assert y.shape == (40,)
    np.testing.assert_allclose(y, np.arange(60, 100) / 99)
    evaluate(models, ValidationMetricEnum.RMSE, scalers)
    eval_x, eval_y, _, _ = recurrent.evaluate.call_args.args
    np.testing.assert_allclose(eval_x, x)
    np.testing.assert_allclose(eval_y, y)


def test_training_and_evaluation_exclude_stock_identifiers_and_dates():
    df = pd.DataFrame({
        'symbole': ['ATB'] * 100,
        'date': pd.date_range('2026-01-01', periods=100),
        'ouverture': np.arange(100, dtype=float) + 1,
        'cloture': np.arange(100, dtype=float) + 2,
        'volume': np.arange(100) + 10,
    })
    original = df.copy()
    model = Mock(spec=IModel)
    model.evaluate.return_value = 0.1
    scalers = train({'GRU': (model, df, df)}, window_size=60)
    scaler = scalers['GRU']
    assert list(scaler.feature_names_in_) == ['ouverture', 'cloture', 'volume']
    x_train, y_train, _ = model.train.call_args.args
    assert x_train.shape == (40, 60, 3)
    assert y_train.shape == (40,)
    np.testing.assert_allclose(y_train, scaler.transform(df[list(scaler.feature_names_in_)])[60:, 1])
    # Column order at evaluation must match training, even if the input changes order.
    reordered = df[['volume', 'symbole', 'cloture', 'date', 'ouverture']]
    result = evaluate({'GRU': (model, df, reordered)}, ValidationMetricEnum.RMSE, scalers)
    assert result == {'GRU': 0.1}
    x_test, y_test, _, _ = model.evaluate.call_args.args
    np.testing.assert_allclose(x_test, x_train)
    np.testing.assert_allclose(y_test, y_train)
    pd.testing.assert_frame_equal(df, original)


def test_get_features_target_from_dataset_regular_case():
    arr = np.array([[i] for i in range(100)])
    X, Y = get_features_target_from_dataset(arr, closeColIdx=0, isArimaContext=False, window=10)
    assert X.shape[0] == 90
    assert X.shape[1:] == (10, 1)
    assert Y.shape[0] == 90

def test_get_features_target_from_dataset_arima_context():
    arr = np.array([[i] for i in range(10)])
    X, Y = get_features_target_from_dataset(arr, closeColIdx=0, isArimaContext=True)
    assert isinstance(X, np.ndarray)
    assert (X == Y).all()

def test_train_with_single_column():
    mock_model = Mock(spec=IModel)
    df = pd.DataFrame({'cloture': range(100)})
    models = {'mock': (mock_model, df, df)}
    scalers = train(models)
    assert 'mock' in scalers
    mock_model.train.assert_called()

def test_train_with_multi_column():
    mock_model = Mock(spec=IModel)
    df = pd.DataFrame({'date': pd.date_range("20210101", periods=100), 'cloture': range(100), 'volume': range(100)})
    models = {'mock': (mock_model, df, df)}
    scalers = train(models)
    assert 'mock' in scalers
    mock_model.train.assert_called()

def test_evaluate_success():
    mock_model = Mock(spec=IModel)
    mock_model.evaluate.return_value = 0.85
    df = pd.DataFrame({'cloture': range(100)})
    models = {'mock': (mock_model, df, df, df)}
    scalers = {'mock': MinMaxScaler().fit(df)}
    result = evaluate(models, ValidationMetricEnum.RMSE, scalers)
    assert result['mock'] == 0.85
    mock_model.evaluate.assert_called()

def test_evaluate_missing_scaler():
    mock_model = Mock(spec=IModel)
    df = pd.DataFrame({'cloture': range(100)})
    models = {'mock': (mock_model, df, df, df)}
    with pytest.raises(KeyError):
        evaluate(models, ValidationMetricEnum.RMSE, {})

def test_plot_evaluation_result_calls_plotting():
    eval_result = {"Model1": 0.9, "Model2": 0.8}
    with patch("matplotlib.pyplot.figure") as mock_figure, \
         patch("matplotlib.pyplot.bar") as mock_bar, \
         patch("matplotlib.pyplot.xlabel") as mock_xlabel, \
         patch("matplotlib.pyplot.ylabel") as mock_ylabel, \
         patch("matplotlib.pyplot.title") as mock_title, \
         patch("matplotlib.pyplot.xticks") as mock_xticks, \
         patch("matplotlib.pyplot.show") as mock_show:
        
        plot_evaluation_result(eval_result, "accuracy")
        
        mock_figure.assert_called_once()
        mock_bar.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_title.assert_called_once()
        mock_xticks.assert_called_once()
        mock_show.assert_called_once()


@patch("matplotlib.pyplot")
def test_plot_stock_graph_calls_plotting(mock_plt):
    dates = np.array([1, 2, 3])
    y_test = np.array([10, 20, 30])
    y_predict = np.array([12, 18, 33])
    with patch("matplotlib.pyplot.figure") as mock_figure, \
         patch("matplotlib.pyplot.plot") as mock_plot, \
         patch("matplotlib.pyplot.xlabel") as mock_xlabel, \
         patch("matplotlib.pyplot.ylabel") as mock_ylabel, \
         patch("matplotlib.pyplot.title") as mock_title, \
         patch("matplotlib.pyplot.legend") as mock_legend, \
         patch("matplotlib.pyplot.show") as mock_show:
        
        plot_stock_graph(dates, y_test, y_predict, "model")        

        mock_figure.assert_called_once()
        mock_plot.assert_called()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_title.assert_called_once()
        mock_legend.assert_called_once()
        mock_show.assert_called_once()

def test_predict_sequence_generation():
    mock_model = Mock(spec=IModel)
    mock_model.predict.side_effect = lambda x: np.array([[0.5]])

    dataset = np.array([[i] for i in range(10)])
    scaler = MinMaxScaler()
    scaler.fit(dataset)

    predictions = predict(mock_model, dataset, scaler, num_days=3, window_size=2)

    # Check correct number of predictions
    assert predictions.shape[0] == 3
    assert mock_model.predict.call_count == 3

@patch("utils.train_test_utils.train")
def test_train_model_trains_and_returns_scaler(mock_train_func):
    mock_model = Mock(spec=IModel)
    model_name = "LSTM"
    dataset = np.array([[i] for i in range(20)])
    old_scaler = MinMaxScaler().fit(dataset)
    window_size = 5
    last_trained_date = datetime.now()

    mock_train_func.return_value = {model_name: old_scaler}
    result_scaler = train_model(mock_model, model_name, dataset, old_scaler, window_size, last_trained_date)

    assert isinstance(result_scaler, MinMaxScaler)
    mock_train_func.assert_called_once()

    
if __name__ == "__main__":
    pytest.main()
