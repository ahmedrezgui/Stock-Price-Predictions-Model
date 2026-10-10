# Verified model comparison — 2026-10-09

Experiments use the supplied BIAT ilboursa raw file, restricted to 2018-01-02
through 2026-10-07 (2,187 observations), version `e2cdd1b2408a383b4dcb`. Prices
have unverified corporate-action adjustments and no authoritative exchange
calendar; h means observed bars. These are retrospective single-stock research
results, not total-return or investable trading performance.

Three seeds (7,19,43), maximum 12 epochs, hidden size 32, one layer, 20-bar
sequences, price+volatility features. Same budgets for LSTM and GRU. MSE/Huber
selection uses mean validation MSE; direction uses Brier. All one-bar selections
chose Huber. Two development walk-forward folds for h=1, one for h=5 and h=20.
Early stopping can produce different realized durations within the same maximum
budget. The fixed final origin span is 2024-12-31 onward; last origins are
2026-10-06 (h=1), earlier for longer horizons. Exact dates/counts are in each JSON.

| Model | h=1 RMSE | h=5 RMSE | h=20 RMSE |
|---|---:|---:|---:|
| Zero return | 0.011523 | 0.028069 | 0.061398 |
| Historical mean | 0.011498 | 0.027828 | 0.059808 |
| ARIMA | 0.011459 | 0.027747 | 0.058778 |
| LSTM, mean of 3 seeds | 0.011462 | 0.027436 | 0.059837 |
| GRU, mean of 3 seeds | 0.011463 | 0.027533 | 0.058409 |

Mean RMSE is the average of separately evaluated models, not an ensemble's RMSE.
One-bar seed std: LSTM 0.0000117; GRU 0.0000234. Five-bar std: LSTM 0.000159,
GRU 0.000117; 20-bar std: LSTM 0.002046, GRU 0.001545. No consistent winner:
LSTM has lower mean five-bar error, GRU lower mean 20-bar error, but the study
is small and overlapping labels reduce effective independent observations.

One-bar historical-frequency probability Brier = 0.244291. LSTM direction mean
Brier = 0.244135, GRU = 0.245059; majority direction accuracy = 57.67%, versus
57.59% LSTM and 56.98% GRU direction heads. Several heads mainly predict the
majority class. Do not infer useful directional skill from raw accuracy alone.
Multitask one-bar mean RMSE: LSTM 0.011460, GRU 0.011464; no reliable improvement
over single-task models is established. Longer-horizon multitask runs were not
included in this bounded experiment.

The networks have 6,466 parameters (LSTM) and 4,866 (GRU), including both defined
heads; unused heads receive no task gradient. Parameter bytes are reported,
**not peak process/GPU memory**. One-bar mean selected-fit durations were about
0.78s LSTM and 1.13s GRU on the available local CUDA device. This includes only
the selected fit, not search/data/report cost. Batch inference latency is timed
after warmup and reported per sample; it is not single-request API latency.
Use report environment metadata before comparing hardware measurements.

## Statistical and economic interpretation

Training ADF p-values: price 0.93193, log-price 0.88634, log-return numerically
underflowed to 0 (report as very small, not mathematical zero). KPSS rejects
price-level stationarity at its lower 0.01 table bound and does not reject return
level stationarity at its upper 0.1 bound. ARCH-LM p=2.55e-17 and squared-return
Ljung–Box lag-10 p=4.43e-19 indicate time-varying volatility. This is compatible
with noisy weakly dependent returns, not proof of easy predictability.

Paired loss tests use HAC and block intervals. One LSTM seed's one-bar DM p versus
zero is 0.0329; others exceed 0.05. This isolated unadjusted result across many
comparisons does not establish robust superiority. See exact fold results and
confidence intervals before making a claim; no multiple-testing correction or
prospective experiment has been completed.

At h=1, next-open simulations with 10 bps per side returned -18.90% to +0.43%
for LSTM regression and +0.03% to +6.26% for GRU regression, compared with gross
buy/hold +72.12% over the same executable span. ARIMA returned -7.48%. Higher
horizons change turnover and holding risk, so compare within horizon, not by
ranking unlike strategies. Unknown adjustments, liquidity, spread and order fills
remain material limitations. No profitable deployable strategy is established.

Full results: [consolidated report](../reports/evaluation_report.md),
[one-bar report](../reports/biat-h1/evaluation_report.md),
[five-bar report](../reports/biat-h5/evaluation_report.md),
[20-bar report](../reports/biat-h20/evaluation_report.md).
