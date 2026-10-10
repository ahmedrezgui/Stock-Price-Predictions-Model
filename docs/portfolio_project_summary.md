# Portfolio project summary

Built a reproducible forecasting service around an existing PyTorch/ilboursa/DVC
project, preserving its downloads and legacy interfaces and leaving LangGraph
unchanged. Audited price-target leakage, interpolated calendar observations,
sequence shape mismatches, scaler refitting and independent artifact loading.

The implemented return pipeline combines strict OHLCV validation, causal feature
groups, horizon-aware label purging, train-only scaling, comparable recurrent
models, train-selected ARIMA and naive baselines. Immutable Parquet manifests and
model bundles link source bytes, feature definitions, split boundaries, model
configuration and code/environment metadata. JSON reports are always local;
MLflow and Azure Storage are optional integrations.

Evaluation separates numerical accuracy, direction, probability calibration,
time-series IC and economic outcomes. It uses repeated seeds, expanding windows,
HAC paired-loss tests, block bootstrap intervals and after-close/next-open execution
with turnover costs. FastAPI loads exact versions and frozen preprocessing, with
readiness, structured errors, tracing and optional local/required hosted API keys.

Verified BIAT one-bar mean RMSE: LSTM 0.011462, GRU 0.011463, ARIMA 0.011459,
zero return 0.011523. Differences are small and architecture ordering varies with
horizon. One-bar recurrent strategy simulations lag gross buy/hold. This is an
engineering/research result, not a claim of market-beating forecasting.

Azure SDK v2 jobs, atomic model registration, Blob adapters, Bicep and Docker
paths are prepared. Cloud creation/deployment was not executed; Docker engine
was unavailable. Student subscription/region/quota and costs require approval
and verification. Data adjustments and a Tunis exchange session calendar remain
external evidence gaps. See implementation_progress.md for actual test outcomes.

## Interview discussion points

1. Why price regression can look accurate while return forecasts barely beat zero.
2. How overlapping h-period labels leak across ordinary chronological boundaries.
3. Why training-only transforms and immutable model/scaler bundles matter in serving.
4. Why sigmoid probability, sign accuracy and calibration are different measurements.
5. Why execution after the information timestamp and turnover costs change results.
6. How fixed search budgets, seed variation, HAC and block bootstrap temper claims.
7. How to design a useful API when calendar/adjustment evidence is missing: explicit
   provenance and null unsupported values instead of invented dates/confidence.
8. Why a student portfolio may use local GPU training, Blob/ML registry and
   scale-to-zero serving instead of an always-on managed endpoint.
