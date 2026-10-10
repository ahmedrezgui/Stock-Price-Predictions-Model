# Evaluation methodology

The supported production research path is `stock_forecasting`, not the legacy
price notebooks. `log_return[t] = log(P[t]/P[t-1])` is historical information;
`target_return[t] = log(P[t+h]/P[t])` and `target_direction[t] = 1(target>0)` are
future labels. The last h direction labels remain missing, never false zeros.
All metrics use decimal log returns. Implied prices use the same price basis as
the return labels. No MAPE is computed on near-zero returns.

## Calendar and data contract

Normalize raw provider bars without interpolation, forward filling or invented
weekends. Exact duplicates collapse; conflicting timestamps, bad OHLC ranges,
nonpositive prices and missing/negative volume fail validation. Zero volume is
retained and reported; economic simulation is unavailable if any execution-span bar has zero volume, until a causal liquidity/fill model is supplied.
Data are exchange-local daily date labels. The current UTC date is excluded
conservatively because the local files have no completion timestamps. The API
rejects data older than `FORECAST_MAX_AGE_DAYS` (14 by default).

With `calendar` (an exchange_calendars identifier) or `session_file` (CSV column
`session`), every expected session must be represented. Gaps fail rather than
turning a 5-session target into 5 available trades. Without a verified calendar,
h means **h observed bars**; API session horizon/date are null. Tunis is absent
from the installed exchange_calendars catalog. BIAT results therefore do not
claim verified trading-session horizons. Supply an authoritative, dated session
file to enable those claims; do not substitute weekdays or a foreign exchange.

The source files have no verified adjustment provenance. Their default policy
is `unverified`; `raw` explicitly requests price returns without dividend
reinvestment. `adjusted` requires adjusted_close and scales all OHLC by
adjusted_close/close, with raw_close retained for volume notional. It is the
operator's responsibility to establish the provider's split/dividend methodology
and point-in-time revision policy. Never describe current BIAT results as total
returns. Symbol mapping/corporate-action reconciliation requires provider records.

## Temporal validation and selection

The default raw-row boundaries are 60% train, 20% validation, 20% test. Sequence
windows end at t and contain only feature rows <=t. Training labels must end
strictly before the first validation origin; validation labels must end strictly
before the first test origin. This purges crossing intervals for h=1,5,20. Test
windows can use preceding historical context. No additional embargo is necessary
under this conservative interval-boundary policy; no randomized split is used.
Scalers fit the union of training input rows once. Target mean/std also fit train
labels only. Validation and serving never fit a transform.

Walk-forward folds end inside the first 80% of data; each uses its own 60/20/20
subsplit, scaler, validation selection and frozen test. Training expands across
folds. Default candidates compare MSE versus Huber on normalized regression
targets, with equal architecture, seeds and epoch budgets for LSTM/GRU. Direction
uses BCEWithLogitsLoss. Multitask uses normalized regression plus 0.5*BCE; selection
uses validation return MSE (direction task: Brier). Average validation performance
across seeds selects the loss, never the best test seed. Each selected seed is
reported separately. Early stopping and LR scheduling read validation only.
The final test is never passed to model selection. Smoke runs and fixed horizon
experiments are engineering checks, not grounds for repeated test-driven tuning.

ARIMA models daily log returns with d=0 and train-only AIC selection over
(0,0,0), (1,0,0), (0,0,1), (1,0,1). Nonconverged candidates are excluded and
warnings saved. Parameters stay frozen; observed returns append through each
forecast origin. Sum h daily forecasts to match cumulative return labels. All
families score exactly the same origins. Neural models use finite recurrent
windows; ARIMA has its own historical state. No model receives future bars.

## Scores and inference

- Regression: MAE/MSE/RMSE/median error, bias, residual std, RÂ², OOS RÂ² versus
  train mean and zero. MASE denominator is mean absolute first difference of
  training h-return labels (a lag-one return-target naive convention).
- Direction: sign for regression, 0.5 threshold for probabilities; accuracy,
  balanced accuracy, precision/recall/F1, confusion matrix and train majority
  accuracy. Undefined denominators return null.
- Probability: only genuine sigmoid or historical-frequency predictions;
  Brier, log loss, ROC-AUC, average precision, ten equal-width-bin ECE. Sigmoid
  outputs are **uncalibrated**, not confidence scores. No return intervals claimed.
- Relationship: Pearson and Spearman; rolling 60-origin rank IC. These are
  single-asset time-series correlations, not daily cross-sectional IC.
- Dependence: squared-error Diebold-Mariano with Bartlett HAC, lag at least h-1
  and floor(n^(1/3)); asymptotic two-sided normal p-value. Circular moving-block
  bootstrap (500 repeats; block=max(h,ceil(n^(1/3)))) gives 95% MAE and direction
  accuracy intervals. Degenerate comparisons return null. Small samples,
  overlapping folds, multiple testing and regime changes limit inference.
- Regimes: realized historical 20-bar volatility relative to its training median,
  and sign of trailing 20-bar momentum, measured at origin. No future-defined
  bull/bear labels enter models. Per-regime metrics are descriptive subsets.

## Economic simulation

A signal after close t enters at open t+1 and exits at open t+h+1. Holdings do
not overlap; intermediate signals are ignored. Execution horizon is separate
from the close-to-close forecasting target. At a scheduled exit, the next eligible
signal can enter. Entry and exit each incur `cost_bps` (default 10); cost totals
are sums of return-unit costs, not currency amounts. Trades are fully invested
long/flat, without leverage; research long/short adds configurable annual borrow
cost and daily rebalancing. Short availability, margin and borrow recalls are
not modeled. Costs approximate commissions/spread/slippage; no order-book model.

Equity compounds simple open-to-open returns less costs. Drawdown includes initial
equity 1. CAGR uses elapsed calendar years; volatility and zero-risk-free Sharpe
use observed intervals per elapsed year. Sortino uses the RMS negative daily
return; Calmar is CAGR/absolute maximum drawdown. Undefined ratios are null.
Win rate and profit factor use completed simulated trades; turnover counts both
sides. Buy/hold is a deliberately stronger **gross** comparator over the identical
open-to-open span. Thresholds are fixed in configuration, never test-optimized.

Sparse sessions, unknown adjustments, illiquidity, survivorship, revised historical
data and lack of executable quotes prevent investable profitability claims.
Forecast accuracy and trading profitability are separate empirical questions.
