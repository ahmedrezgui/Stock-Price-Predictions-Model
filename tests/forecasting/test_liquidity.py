import numpy as np
import pytest
from stock_forecasting.backtesting import backtest


def test_zero_volume_exit_does_not_select_trades_with_future_information(bars):
    bars.loc[36, "volume"] = 0
    with pytest.raises(ValueError, match="liquidity"):
        backtest(bars, np.arange(30, 60), np.ones(30), horizon=5)
