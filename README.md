# Stock forecasting with PyTorch

Compare LSTM, GRU, ARIMA and simple baselines on Tunisian stock returns, then
train versioned candidates and serve predictions through FastAPI.

## Start with the benchmark

Open [Models_benchmark.ipynb](notebook/Models%20benchmark/Models_benchmark.ipynb)
for the complete workflow and saved outputs. Read the
[benchmark report](reports/benchmark/benchmark_report.md) for the model-choice
rationale, metrics, per-stock results and figures.

The experiment covers **BIAT, ATB, ARTES, NAKL and CIL**, using available history
from 2010 through October 2026. All models for a stock use the same forecast
origins. All stocks share calendar boundaries: training before 2022, validation
in 2022?2023 and test from 2024. Available trading rows can differ across stocks.
Future labels crossing split boundaries are purged; scalers fit training inputs only.

LSTM and GRU use the existing public model classes and shared training engine.
They receive matching features, sequence length, hidden size, optimizer settings,
three seeds (7, 19, 43), a 50-epoch maximum and patience 10. Early stopping restores
the best validation checkpoint, so actual epoch counts differ. The notebook
requires CUDA for all 60 neural fits and neural predictions; ARIMA runs on CPU
with training-only AIC order selection. Matching neural budgets does not imply
equal parameter counts or equal compute costs across architectures.

### Model choice

Lower relative RMSE is better; 1 is the historical-mean baseline. Scores average
seeds within each stock, then weight the five stocks equally.

| Return model | Validation relative RMSE | Test relative RMSE |
|---|---:|---:|
| Historical mean | 1.000000 | 1.000000 |
| ARIMA | 1.002528 | 0.996132 |
| LSTM | 0.997116 | 1.000522 |
| GRU | 1.000840 | 0.997867 |

**LSTM is the provisional validation-selected candidate.** Its validation advantage
is only 0.288%, and it beats the mean baseline on two of five stocks. ARIMA has the
lowest test error; LSTM slightly underperforms the baseline on test. These results
justify further evaluation of LSTM, not declaring it the production winner.
Freeze the configuration and compare against ARIMA and simple baselines on new,
unseen observations before making a deployment decision.

The report includes forecasts, training curves, confusion matrices, probability
and strategy diagnostics: one Markdown file and 31 PNGs. Displayed tables persist
in notebook outputs; full predictions live in kernel memory and require rerunning
after restart. Benchmarking does not save or promote a serving model.

## Local setup

Use Python 3.11 or newer and a CUDA-enabled PyTorch environment for the notebook.
The notebook checks CUDA availability and stops rather than falling back to CPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Raw datasets are not committed. Obtain the five CSVs in `data/raw/` through the
ilboursa download script or restore your local datasets before rerunning the
notebook. The report records each source checksum and actual date coverage.
Choose the project environment as the notebook kernel and run its cells in order.
The viewing cell can be rerun without repeating training.

## Routine training and API

```powershell
python -m scripts.train_model --config configs/training.json --symbols BIAT
python -m uvicorn stock_forecasting.api:app --host 127.0.0.1 --port 8000
```

Routine training uses its own configuration; the benchmark explicitly overrides
history, epoch budget and device. Review the selected configuration before
training a deployment candidate. The API exposes `/health`, `/ready`, `/models`
and `/v1/predict`, with interactive documentation at `/docs`. Serving uses saved
model bundles and frozen preprocessing, not the benchmark's in-memory predictions.

## MLflow experiment tracking

Install tracking once in the project environment:

```powershell
python -m pip install -e ".[dev,tracking]"
```

Run these commands from the repository root to track routine training locally:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
$env:MLFLOW_EXPERIMENT_NAME = "stock-return-forecasting"
python -m scripts.train_model --config configs/training.json --symbols BIAT
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 and select `stock-return-forecasting`. The notebook uses
this same database by default, with an absolute path so its working directory
does not matter. The saved notebook was rerun with tracking enabled: the local
database contains 90 benchmark runs (five stock parents and 85 model/baseline
children), plus a routine BIAT training check. That candidate was correctly
recorded as rejected by the existing baseline gate. Training verification used
an isolated `artifacts/tracking-validation/` directory. These local tracking files
are not committed; a fresh clone must rerun to create its own experiment history.

Both command-line entry points accept `--mlflow-uri` and `--mlflow-experiment`.
Explicit arguments take priority over environment variables. Python functions
`benchmark_stock` and `train_candidate` accept `mlflow_uri` and
`mlflow_experiment`. Outside the notebook, tracking is disabled when no URI is
configured. An enabled but unavailable tracker raises an error; it does not
silently discard evidence. For a shared tracking server, set the same variables
to its URI and configure its authentication through MLflow's environment options.

| Information | Where to find it |
|---|---|
| Model settings, seed, maximum epochs, feature version | Parameters |
| Source CSV SHA-256 and normalized dataset version | Parameters and `metadata/dataset.json` |
| Feature order, date boundaries and purge/scaler details | `metadata/inputs.json` |
| Validation/test RMSE, MAE, direction and probability scores | Metrics such as `validation.regression.rmse` and `test.probability.brier` |
| Per-epoch loss and learning rate | `training.*` metric history and `metadata/history.json` |
| Git revision, dirty working tree, package versions | Tags and `metadata/environment.json` |
| Benchmark report and figures | Stock run's `reports/` artifacts after the export cell |
| Routine model weights, preprocessing and checksums | `models/<version>/` artifacts |
| Candidate decision, baseline scores and refit policy | Tags and `metadata/candidate.json` |

Each notebook stock run groups child runs for LSTM/GRU tasks and seeds, ARIMA,
and the simple baselines. The full model configuration is recorded; neural-only
settings do not apply to ARIMA or constant baselines. `actual_device` identifies
the device used by fitted models. Benchmarking keeps its local output to one
Markdown report and figures; MLflow JSON artifacts are created through temporary
files. It does not save benchmark weights. A stock run can finish before figure
export; `report_exported=true` confirms the export cell uploaded its artifacts.

Routine training records validation scores without evaluating the research test
holdout. A production refit stores both candidate and refit bundles; validation
scores belong to the development candidate, not the refitted weights. An unchanged
cached candidate creates a new run tagged `status=unchanged` and reuses the model
version; its recorded history is the original training history, not a new fit.
Tracking uploads happen before local promotion. Failed fits or uploads mark the
run `FAILED`; rejected candidates are successful runs with `status=rejected`.

Dataset versions use the same normalized-data/metadata hash in both workflows.
Different history cutoffs legitimately produce different versions. Raw CSVs are
not uploaded to MLflow; retain them through DVC/Blob storage to reproduce a run.
Undefined metrics are omitted from numeric charts and retained as null in JSON.
Metrics are logged explicitly using the [MLflow client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html).

`mlflow.db` and `mlruns/` stay out of Git. Keep both when backing up local tracking.
The training workflow enables SQLite tracking and retains the database and
artifacts with its GitHub Actions outputs; each job has a separate local store.
This is not yet a shared hosted tracking service. Azure jobs use the tracking URI
provided by Azure ML unless explicitly overridden; no cloud job was submitted
as part of implementing this step.

## Validation and remaining work

```powershell
python -m pytest tests/forecasting -q
python -m ruff check stock_forecasting tests/forecasting
```

Provider prices do not have verified corporate-action adjustments or exchange
calendars. Horizons count observed bars. Strategy plots include simplified costs
and execution assumptions and do not establish trading profitability. Previously
inspected historical test periods are retrospective evidence.

Azure integration, candidate promotion and monitoring are separate from this
benchmark milestone. A successful benchmark is not proof of a verified live Azure
deployment or performance-monitoring service. The `docs/` directory contains local
project guides and is intentionally excluded from Git; it is absent from a fresh
clone. The existing research CLI and ETL dependencies remain available.
