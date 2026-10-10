# Architecture audit — 2026-10-09

This describes the repository **before this mission's changes**. The working tree
already contained a PyTorch migration, downloader changes, tests, documentation,
and removed recommender files. Those edits are not assumed to be ours. No Azure
resources were inspected; Azure CLI was not on PATH. No cloud operation was run.

## Current architecture and components searched

`src/prediction_model/models/{LSTMModel,GRUModel,ARIMAModel,IModel}.py`
contains the model abstraction. LSTM is a two-layer, 64-unit PyTorch network
with a dense head; GRU has 50 units. Both run 50 epochs with Adam/MSE, automatic
CUDA selection, no validation stopping, seed control or checkpoint selection.
ARIMA uses statsmodels with fixed (2,1,5), flattens its input and forecasts a
single uninterrupted future path. No trained .pt/.pth/.pkl/.h5 files were found
outside the virtual environment. Model and scaler handlers use pickle/joblib,
separate local directories, or independently registered Azure ML assets, pinned
to version 1. Missing/corrupt remote artifacts are broadly swallowed.

`scripts/download_data.py` and `download_data_playwright.py` ingest ilboursa
Tunisian exchange CSVs. The existing requests adapter explicitly detects security
challenges and invalid CSV responses. Retain it; do not bypass access controls.
31 raw CSVs are locally available, some spanning 2010–2026. DVC pointers cover
additional symbols. French columns use semicolon delimiters and decimal commas.
No adjusted-close column, corporate-action ledger, publication timestamps or
authoritative exchange calendar accompanies the files. Adjustment methodology
cannot be verified from the CSVs. Sparse/inactive instruments exist.

`scripts/process_data.py` inserts **all calendar dates** and linearly interpolates
OHLCV between past and future observations. These processed files are unsuitable
for leakage-safe experiments. Rebuild research inputs directly from raw files.
`utils/train_test_utils.py` splits chronologically, fits MinMaxScaler on the
supplied training frame, and targets the following closing price. It accepts
arbitrary numeric columns, including future targets if supplied by a notebook.
It can refit a scaler while continuing an existing model. Evaluation uses a
different default sequence length. Recursive prediction supplies 2-D inputs to
3-D recurrent models and cannot synthesize future multi-feature observations.

`scripts/train_model.py` incrementally trains per-stock price models, saves
model/scaler separately, uses a threshold evaluation and email notification.
The benchmark notebook contains return and direction columns but passes its frame
to the old closing-price utility, exposing target leakage; its ARIMA entry also
uses the training frame as test data. The ADF notebook tests prices/differences
and overstates rejection as proof of stationarity. Existing plots show prices
and aggregate RMSE/MAPE. No complete return benchmark/report engine was found.

## API and inference convention

`src/web/back/main.py` is FastAPI: GET `/companies`, GET `/stock/{company}`,
POST `/rag`. It initializes the agent graph at import. The shared
`src/services/stock_data_service.py` powers the web and RAG clients. `/stock`
appends predicted prices as synthetic OHLC rows; it requires model training date
to equal the latest data date. Dates are incremented by calendar days. There is
no enforced after-close convention, calendar, adjustment provenance, model
version selection, readiness endpoint or frozen feature manifest.
`src/endpoint/score.py` is a separate Azure scoring adapter using `data` arrays;
it also feeds 2-D scaled data to recurrent prediction. Preserve legacy routes
and their contracts; introduce a standalone versioned forecasting service so
Part A can run without agent credentials. Do not alter the LangGraph system.

## Azure, dependencies, CI and observability

`environment*.yml`, `requirements.txt`, frontend/backend Dockerfiles,
`.github/workflows/{ci,cd,etl,train,news,stock_data}.yml`, DVC and `.env.example`
exist. Azure ML SDK v2 and DefaultAzureCredential are already used by handlers.
The train workflow actually runs training on the GitHub runner, despite its
Azure-job label. CD pushes images and changes App Service on main pushes.
The legacy deploy script creates a paid managed endpoint at **import time**,
despite a comment saying it is unused. These automatic cloud mutations need
explicit manual gating. Keys/subscription placeholders exist in the example;
new cloud paths should use managed identity/RBAC and parameterized identifiers.
Logging is print/basic logging. No experiment lineage, atomic bundle verification,
financial backtest, calibration or drift system was found. Existing tests include
model, downloader, utilities, services, API and unrelated agent tests.

## Baseline verification

Python 3.12.8 in `.venv`; PyTorch/pandas/statsmodels/scikit-learn/pyarrow/FastAPI
available; MLflow initially absent. Focused baseline test run: **48 passed,
6 failed**. Failures: GRU input validation disabled; train sequence defaults
disagree with tests (1 versus 60); two tests pass duplicate close-column arguments.
These are existing failures, not new return-pipeline regressions.

## Target architecture and prioritized plan

1. Preserve existing downloads, DVC, PyTorch work, legacy web/agent interfaces.
   Make immediate safe correctness fixes; prohibit import-time cloud changes.
2. Add an independent `stock_forecasting` package with strict raw normalization,
   causal feature groups, aligned future return/direction labels, immutable local
   or Blob snapshots, train-only scaling and purged chronological folds.
3. Comparable configurable LSTM/GRU encoders, classification and normalized
   multitask heads; separate statsmodels ARIMA state updates and naive baselines.
4. Validation-only selection; one final holdout; dependent-loss statistics,
   next-open execution backtests, local tracking and optional MLflow, reports.
5. One model bundle contains weights, preprocessing, feature/target definitions,
   dataset lineage, checksums and metadata. A standalone FastAPI service uses it.
6. SDK v2 jobs/registry/scoring, local-first Docker and parameterized infrastructure;
   consult current Microsoft documentation. No provisioning/deployment without
   explicit approval, credentials, region and budget verification.
7. Run meaningful tests and real-data bounded benchmarks. State missing adjustment
   and exchange-calendar evidence explicitly; never call observed rows verified
   trading sessions or unadjusted returns total returns. Document blocked claims.
