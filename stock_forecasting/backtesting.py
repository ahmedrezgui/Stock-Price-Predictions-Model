"""After-close decisions enter next open; fixed h-bar holding, no overlap."""

import numpy as np
import pandas as pd

from .evaluation import safe_ratio


def backtest(
    frame: pd.DataFrame,
    origins,
    signals,
    horizon=1,
    threshold=0.0,
    cost_bps=10.0,
    long_short=False,
    borrow_bps_year=300.0,
):
    if len(origins) != len(signals) or cost_bps < 0 or horizon < 1 or threshold < 0:
        raise ValueError("Invalid backtest inputs")
    if not np.isfinite(signals).all() or np.any(np.diff(origins) <= 0):
        raise ValueError("Signals must be finite with unique chronological origins")
    start = int(origins[0]) + 1
    stop = min(int(origins[-1]) + horizon + 1, len(frame) - 1)
    if (frame.volume.iloc[start : stop + 1] <= 0).any():
        raise ValueError("Zero-volume execution bars require a liquidity/fill model; simulation is unavailable")
    position = np.zeros(stop - start)
    turnover = np.zeros(stop - start)
    trades = []
    next_entry = start
    for origin, signal in zip(origins, signals):
        entry, exit_ = int(origin) + 1, int(origin) + 1 + horizon
        side = 1 if signal > threshold else (-1 if long_short and signal < -threshold else 0)
        if side == 0 or entry < next_entry or exit_ > stop:
            continue
        position[entry - start : exit_ - start] = side
        turnover[entry - start] += 1
        turnover[exit_ - start - 1] += 1
        trade_daily = side * (
            frame.open.iloc[entry + 1 : exit_ + 1].to_numpy() / frame.open.iloc[entry:exit_].to_numpy() - 1
        )
        trade_daily[0] -= cost_bps / 10000
        trade_daily[-1] -= cost_bps / 10000
        if side < 0:
            trade_daily -= borrow_bps_year / 10000 / 252
        trade_return = np.prod(1 + trade_daily) - 1
        trades.append(
            {
                "decision": str(frame.timestamp.iloc[origin]),
                "entry": str(frame.timestamp.iloc[entry]),
                "exit": str(frame.timestamp.iloc[exit_]),
                "side": side,
                "net_trade_return": float(trade_return),
            }
        )
        next_entry = exit_
    market = frame.open.iloc[start + 1 : stop + 1].to_numpy() / frame.open.iloc[start:stop].to_numpy() - 1
    costs = turnover * cost_bps / 10000 + (position < 0) * borrow_bps_year / 10000 / 252
    gross = position * market
    net = gross - costs
    if not len(net) or (net <= -1).any():
        raise ValueError("No executable periods or insolvency")
    equity = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1, equity])[1:]
    drawdown = equity / peak - 1
    years = max((frame.timestamp.iloc[stop] - frame.timestamp.iloc[start]).days / 365.25, 1 / 365.25)
    annual_factor = len(net) / years
    cagr = float(equity[-1] ** (1 / years) - 1)
    volatility = float(np.std(net, ddof=1) * np.sqrt(annual_factor)) if len(net) > 1 else 0
    downside = np.sqrt(np.mean(np.minimum(net, 0) ** 2)) * np.sqrt(annual_factor)
    trade_returns = np.array([t["net_trade_return"] for t in trades])
    benchmark = np.cumprod(1 + market)
    metrics = {
        "gross_cumulative_return": float(np.prod(1 + gross) - 1),
        "net_cumulative_return": float(equity[-1] - 1),
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_risk_free": safe_ratio(np.mean(net) * annual_factor, volatility),
        "sortino_zero_target": safe_ratio(np.mean(net) * annual_factor, downside),
        "max_drawdown": float(drawdown.min()),
        "calmar": safe_ratio(cagr, abs(drawdown.min())),
        "win_rate": float(np.mean(trade_returns > 0)) if len(trades) else None,
        "average_trade_return": float(np.mean(trade_returns)) if len(trades) else None,
        "profit_factor": safe_ratio(trade_returns[trade_returns > 0].sum(), -trade_returns[trade_returns < 0].sum()),
        "turnover": float(turnover.sum()),
        "number_of_trades": len(trades),
        "transaction_cost_total_return_units": float(costs.sum()),
        "exposure": float(np.mean(abs(position))),
        "buy_hold_gross_return": float(benchmark[-1] - 1),
        "net_minus_buy_hold_gross": float(equity[-1] - benchmark[-1]),
        "holding_period_bars": horizon,
        "signal_policy": "nonoverlapping; enter next open, exit h opens later",
        "annual_periods_observed": annual_factor,
    }
    curve = pd.DataFrame(
        {
            "timestamp": frame.timestamp.iloc[start + 1 : stop + 1].to_numpy(),
            "position": position,
            "gross_return": gross,
            "net_return": net,
            "equity": equity,
            "buy_hold": benchmark,
            "drawdown": drawdown,
            "cost": costs,
        }
    )
    return metrics, curve, trades
