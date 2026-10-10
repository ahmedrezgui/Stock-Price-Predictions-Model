import numpy as np
import pytest

from stock_forecasting.backtesting import backtest
from stock_forecasting.evaluation import block_interval, diebold_mariano, evaluate


def test_metrics_known_values_and_undefined():
    y = np.array([-0.1, 0.1, -0.2, 0.2])
    result = evaluate(y, {"return": np.zeros(4), "probability": np.full(4, 0.5)}, y)
    assert result["regression"]["mae"] == pytest.approx(0.15)
    assert result["regression"]["rmse"] == pytest.approx(np.sqrt(0.025))
    assert result["regression"]["oos_r2_zero"] == pytest.approx(0)
    assert result["probability"]["brier"] == 0.25
    assert result["probability"]["roc_auc"] == 0.5
    assert result["direction"]["accuracy"] == 0.5
    assert result["relationship"]["pearson"] is None
    constant = evaluate(np.zeros(10), {"return": np.zeros(10), "probability": np.zeros(10)}, np.zeros(10))
    assert constant["regression"]["r2"] is None
    assert constant["regression"]["mase"] is None
    assert constant["probability"]["roc_auc"] is None
    with pytest.raises(ValueError):
        evaluate(y, {"probability": np.full(4, 2)}, y)


def test_dependence_statistics():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 200)
    a = diebold_mariano(y, y, np.zeros(200), horizon=20)
    assert a["statistic"] < 0 and a["p_value"] < 0.05 and a["hac_lags"] >= 19
    assert diebold_mariano(y, y, y)["p_value"] is None
    assert block_interval(np.ones(100), 5) == [1, 1]


def test_backtest_next_open_costs_and_no_overlap(bars):
    origins = np.arange(30, 60)
    signal = np.ones(len(origins))
    free, curve, trades = backtest(bars, origins, signal, horizon=5, cost_bps=0)
    paid, paid_curve, _ = backtest(bars, origins, signal, horizon=5, cost_bps=10)
    assert all(t["entry"] > t["decision"] for t in trades)
    assert all(a["exit"] <= b["entry"] for a, b in zip(trades, trades[1:]))
    assert paid["net_cumulative_return"] < free["net_cumulative_return"]
    assert paid["turnover"] == 2 * paid["number_of_trades"]
    assert paid["transaction_cost_total_return_units"] == pytest.approx(paid["turnover"] * 0.001)
    np.testing.assert_allclose(paid_curve.equity, np.cumprod(1 + paid_curve.net_return))
    assert paid["max_drawdown"] <= 0
    # Same-bar close cannot affect next-open execution P&L.
    changed = bars.copy()
    changed.loc[origins, "close"] *= 100
    other, _, _ = backtest(changed, origins, signal, horizon=5, cost_bps=0)
    assert other["net_cumulative_return"] == free["net_cumulative_return"]


def test_flat_strategy_no_trades(bars):
    result, _, _ = backtest(bars, np.arange(30, 60), np.zeros(30))
    assert result["number_of_trades"] == 0 and result["net_cumulative_return"] == 0
    assert result["sharpe_zero_risk_free"] is None
    assert result["profit_factor"] is None
