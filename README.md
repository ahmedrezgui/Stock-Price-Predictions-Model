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
Training never changes the active serving model. Tracking uploads precede registry registration. Failed fits or uploads mark the
run `FAILED`; rejected candidates are successful runs with `status=rejected`.

Dataset versions use the same normalized-data/metadata hash in both workflows.
Different history cutoffs legitimately produce different versions. Raw CSVs are
not uploaded to MLflow; retain them through DVC/Blob storage to reproduce a run.
Undefined metrics are omitted from numeric charts and retained as null in JSON.
Metrics are logged explicitly using the [MLflow client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html).

`mlflow.db` and `mlruns/` stay out of Git. Keep both when backing up local tracking.
The training workflow uses the shared `MLFLOW_TRACKING_URI` repository variable
when configured, otherwise SQLite, and retains its local evidence in GitHub
Actions outputs. SQLite jobs have separate local stores.
This is not yet a shared hosted tracking service. Azure jobs use the tracking URI
provided by Azure ML unless explicitly overridden; no cloud job was submitted
as part of implementing this step.

## Deployment protection and Model Registry (BIAT)

BIAT is the first supported release target, for one-observation-ahead return
forecasts. Its long history makes it suitable for building this workflow; it does
not establish that LSTM is best. The existing benchmark MLflow runs contain
scores and figures, **not serving weights**. Routine training now registers its
verified custom model bundles under `stock-BIAT-return-h1`. These bundles use
our existing loader/API; they are not MLflow pyfunc serving packages.

The registry distinguishes two aliases:

- `candidate`: the latest registered candidate, including its development status.
- `deployed`: set only after deployment verification confirms the exact model is
  answering requests. Training and successful evaluations never set this alias.

Training no longer promotes models into `active.json`. Development validation
returns `candidate` or `rejected`; a release decision requires separate evidence.
Rejected development candidates remain registered for audit and may be evaluated
prospectively; registration alone is never approval. Repeated cached training
reuses the same registry version so it does not reset the registration cutoff.
Local `registry.promote` remains a development/testing utility and cannot bypass
release packaging's MLflow check.

### Frozen release policy

The reviewable rules live in [configs/release_policy.json](configs/release_policy.json).
The default policy is:

| Requirement | Replacement release | First deployment (`--bootstrap`) |
|---|---|---|
| Eligible observations | At least 120 paired outcomes | At least 120 matured outcomes |
| RMSE | At least 1% better than deployed | At least 1% better than frozen mean baseline |
| MAE | No worse than deployed | No worse than frozen mean baseline |
| Mean baseline | Must also beat it | Used as the comparator |
| Input freshness | Latest completed observation within 14 days | Same |
| Approval lifetime | 24 hours | Same |

Both models use **the exact same origins and outcomes**, with their own frozen
features and scalers. No fitting occurs during comparison. The baseline is the
candidate's saved training-target mean, never a mean fitted on evaluation data.
Changing architectures is allowed; changing symbol, horizon, task, adjustment
basis or calendar definitions is blocked.

Eligible origins must be later than every model's training-label and
validation-label cutoffs, bundle creation time, registry registration time, and
policy freeze time. Current-day data and immature labels are excluded. This
conservative prospective rule also excludes the already-inspected benchmark.
Missing dates are not manufactured; the horizon still counts observed bars.
The 120-observation rule is an operational minimum, not statistical proof of
superiority or trading profitability.

**There is currently no deployed model.** Normal replacement evaluations must
block. Bootstrap is explicit and still requires fresh evidence; it does not
create a pretend deployed model or waive thresholds. Local verification registered BIAT version 1 and
confirmed that both replacement and bootstrap checks block with zero eligible
observations; no deployed alias was assigned. A newly registered model
cannot pass using historical benchmark outcomes. Freeze the candidate and wait
for at least 120 eligible new observations. Registering a newly trained version
starts its own prospective evidence period.

Automatic training jobs no longer use `--refit-production`. An explicit refit
remains available, but all labels it uses become ineligible for release evaluation.
Never refit approved weights before release: the approval pins their checksums.

### Local use

Activate the project environment and run from the repository root:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
python -m scripts.train_model --config configs/training.json --symbols BIAT
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open MLflow's **Models** page, select `stock-BIAT-return-h1`, and note the exact
numeric candidate version. The following uses version 1 as an example:

```powershell
python -m stock_forecasting.release_gate --mlflow-uri sqlite:///mlflow.db --name stock-BIAT-return-h1 --version 1 --data data/raw/BIAT.csv --bootstrap
```

A blocked decision exits with code 1 and records the reasons in a `release_gate`
MLflow run. Its artifacts include `metadata/gate.json`; scored comparisons also
include `metadata/paired_predictions.json`. The run records the policy hash,
model checksums, source CSV checksum, eligible dates/count, metrics and expiry.
Artifact integrity is checked against an immutable MLflow run parameter. An
unreachable registry or service, invalid predictions, insufficient evidence,
corrupt bundles, or missing provenance blocks release.

Only after a passing gate, use its run ID to package the exact approved bundle:

```powershell
python -m stock_forecasting.release --mlflow-uri sqlite:///mlflow.db --gate-run-id <GATE_RUN_ID> --data data/raw --output deployment
```

Packaging fetches evidence from MLflow, rejects superseded/expired gates, verifies
that the policy, source CSV, candidate and deployed version have not changed,
and includes only the approved model. A hand-edited local approval JSON or local
active-model entry is insufficient. Output directories must be new.

### Later Azure deployment

The release workflow now evaluates before packaging, image publication and Azure
changes. It takes an exact candidate registry version, a workflow run containing
fresh CSV data, and an explicit bootstrap flag. It rechecks approval immediately
before deployment and confirms the model served by `/v1/predict` before assigning
`deployed`. Subsequent gates check the deployed endpoint as well as registry
lineage. `FORECAST_API_KEY` supplies the API key; it is not stored in gate artifacts.

Release jobs require a **shared** MLflow service configured with the repository
variable `MLFLOW_TRACKING_URI` and, where needed, secret `MLFLOW_TRACKING_TOKEN`.
A laptop SQLite database is not a shared registry for GitHub runners. Configure
artifact storage and authentication for your chosen MLflow host in the Azure
setup stage. The current workflow targets the existing App Service container
path; Azure ML managed endpoints will need their corresponding deployment adapter.
The older direct Azure ML `deploy` command is disabled so it cannot bypass this
gate. The legacy frontend/backend web workflow is a separate application, not the
forecasting-service release route.
No Azure resource or live deployment is created by the local implementation.

Keep all deployment writers in the workflow's single concurrency group and use
protected branches plus the `azure-approved` environment. MLflow alias updates
are not atomic compare-and-swap operations; administrators must not edit aliases
or artifacts during a release. Restrict registry/artifact writes to trusted
identities. This gate protects the release path, not against an administrator
who rewrites both the registry and CI rules. If deployment succeeds but endpoint
verification or alias recording fails, the workflow fails and later releases
block until the actual serving state is reconciled; automatic Azure rollback is
not implemented in this step.

Registry operations use the [MLflow Model Registry API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html).

## Step 4: prediction monitoring

The API now saves each issued forecast before returning it and includes a
`forecast_id` in the response. The Azure scoring adapter uses the same store.
SQLite keeps the forecast, model version, stock, task, horizon, original close,
input/training dataset versions and issue time. No additional database server is
needed locally. Repeating the same forecast returns the same ID and does not
increase its weight in the metrics. Conflicting inputs or predictions for that
same model version and origin are rejected; investigate the source revision.
If recording fails, the API returns 503 instead of an unrecorded forecast.
Direct research calls to `predict_bundle` are not recorded unless `monitor_path`
is provided; benchmark predictions are not retroactively treated as live forecasts.

### Run locally

Use the same database path for serving and monitoring. From the repository root,
with the virtual environment activated, set this in **both** terminals:

```powershell
$env:FORECAST_MONITOR_DB = "$env:LOCALAPPDATA\StockForecast\forecasts.sqlite3"
```

This keeps the active database outside the project's OneDrive folder. Without
the setting, the default API stores it in `artifacts/monitoring/forecasts.sqlite3`.
Custom model roots use a sibling `monitoring` folder, so use an explicit path
when running the monitoring command against a custom API installation.

Start your existing API and request predictions as usual:

```powershell
python -m uvicorn stock_forecasting.api:app --host 127.0.0.1 --port 8000
```

After refreshing the raw CSV files, reconcile outcomes and update the report:

```powershell
python -m stock_forecasting.monitoring --data data/raw --window 30 --min-observations 20 --mlflow-uri sqlite:///mlflow.db
```

There is one report: `reports/monitoring/monitoring_report.md`, overwritten each
run. The database and generated report are ignored by Git. Without `--mlflow-uri`
or `MLFLOW_TRACKING_URI`, local monitoring still works. When enabled, MLflow gets
one `prediction_monitoring` run per model identity, with rolling metric history,
counts, exact matched forecasts/outcomes and dataset versions in
`metadata/monitoring.json`, plus the Markdown report. It does not create new model
versions or change registry aliases.

Forecast capture is automatic; reconciliation runs when this command runs.
Schedule it daily **after the CSV refresh** on the machine owning the database.
For Windows Task Scheduler, use the full path to `.venv\Scripts\python.exe` as
the program, set this repository as **Start in**, and use these arguments
(replace the database path with your actual absolute path):

```text
-m stock_forecasting.monitoring --database C:\Users\rezgu\AppData\Local\StockForecast\forecasts.sqlite3 --data data/raw --mlflow-uri sqlite:///mlflow.db
```

No scheduled task or cloud resource is installed automatically. A GitHub runner
cannot read your laptop's forecast database. Refresh CSVs atomically (write a
temporary file, then replace the destination) to avoid readers seeing partial
downloads. Configure the scheduler to prevent overlapping reconciliation jobs.

### Matching rules and metrics

Only completed daily observations are used; the current UTC date is excluded,
as in training and inference. An unverified horizon of 5 means the fifth observed
bar after the forecast origin, not five calendar days. Verified session forecasts
also require the target date saved at issue time. No missing day is filled.
The actual return is `log(target_close / origin_close)` using the forecast's
original price-adjustment policy.

Outcomes stay pending until enough completed bars exist. Missing sources,
revised origin prices, revised matched outcomes, calendar mismatches and outcomes
dated before issuance are excluded and explained in the report. Matched outcomes
are rechecked on subsequent runs; original evidence is preserved. The command
exits with code 1 for data issues, so a scheduler can surface failures. Pending
outcomes and a short evaluation window are normal states, not command failures.

| Metric | Meaning |
| --- | --- |
| RMSE | Typical log-return error, with greater weight on large errors; lower is better. |
| MAE | Average absolute log-return error; lower is better. |
| Bias | Average predicted minus actual return; positive means overprediction. |
| Zero-return RMSE | Error of always predicting no price change, over the same window. |
| Direction accuracy | Fraction predicting the correct positive versus nonpositive return. |
| Brier score | Average squared error of the probability of an up move; lower is better. Only available for probability outputs. |
| n / sufficient_evidence | Matched origins in the window; defaults to needing 20 of the last 30. This is a descriptive reporting minimum, not the release gate's evidence threshold. |

Each stock, exact model version, task, horizon and price basis has its own rolling
window. Direction-only models have no return-error metrics. Zero actual return
counts as nonpositive. Returns are dimensionless natural-log returns, not price
errors or percentage-point errors. Longer overlapping horizons produce correlated
errors; their observation count does not imply independent samples.

Daily CSV dates cannot prove that same-day issuance preceded the market close.
These cases are counted as `same_day_issue_n`; monitoring is daily-bar error
tracking, not verified intraday trading performance. Unverified observed-bar
horizons also cannot detect a missing exchange session. The report shows the
latest outcome date: rerunning it with stale CSVs does not produce fresh evidence.
Monitoring does not replace or automatically relax step 3's release protection.

### Storage for later deployment

Production serving requires an explicit `FORECAST_MONITOR_DB` path. The release
workflow passes the repository variable of that name to the service; setting a
path alone does not provision durable storage. The current store supports one
host with a durable local disk and concurrent local processes. Keep backups using
SQLite's backup facility or copy the file while all writers are stopped.

Do not place the active SQLite database on a shared network filesystem; see
[SQLite's network filesystem guidance](https://www.sqlite.org/useovernet.html).
Azure App Service and Azure ML deployment need a suitable persistent monitoring
store and scheduled reconciliation connected to refreshed data before launch.
Their container writable layer is not a durable deployment solution; see
[Azure container storage configuration](https://learn.microsoft.com/en-us/azure/app-service/configure-custom-container).
For those managed services or multiple replicas, adapt the small storage layer
to a shared database during the Azure deployment step. Do not point multiple
replicas at independent local files or a network-mounted SQLite file.

## Market-wide local LSTM training and DVC datasets

Run the complete local preparation and candidate-training pipeline:

```powershell
python -m stock_forecasting.market
```

The default universe combines `utils/constants.py`, downloaded `data/raw/*.csv`
files and existing raw DVC pointers. This includes stocks temporarily commented
out of the downloader list and reports missing previously tracked downloads.
To validate/cache datasets without training, use `--prepare-only`. For a selected
batch use `--symbols BIAT ATB`. `--as-of YYYY-MM-DD` freezes the exclusive
completed-day cutoff. The command uses the local `sqlite:///mlflow.db` registry;
`--mlflow-uri` overrides it. Azure registry compatibility remains a separate task.

The steps are:

1. Preserve the raw CSV bytes and validate each symbol independently. Normalize
   French column names, decimal commas, dates and ordering; remove identical
   duplicate rows. Conflicting duplicates, missing/invalid prices or volumes and
   invalid OHLC ranges block the symbol. No prices or absent sessions are filled.
2. Save canonical, **unscaled** daily bars as `data/processed/<SYMBOL>.csv`.
   Persist hashes and per-symbol validation outcomes in
   `data/processed/manifest.json`. Globally fitting a scaler here would leak future
   data; each model instead fits its scaler on its training input rows only.
3. Run local `dvc add` on downloaded raw CSVs, valid processed CSVs and the
   manifest. This updates/cache-stores their `.dvc` pointers without a Git commit
   or Azure push. Invalid raw files are still versioned as evidence.
4. Exclude stale histories (more than 14 calendar days) and insufficient samples
   (fewer than 252 train, 60 validation or 60 holdout samples after feature warmup
   and label purging). Valid excluded histories remain processed for research.
5. Train one LSTM return candidate per eligible symbol, sequentially on CUDA.
   `configs/training.json` uses history from 2010, maximum 50 epochs, patience 10
   and seed 7. The chronological split is 60% training / 20% validation / 20%
   untouched holdout, separately within each stock's available history. This is
   candidate training, not a new common-calendar architecture benchmark.
6. Log models, fitted scalers, raw/processed dataset lineage, validation metrics
   and learning curves to MLflow. The processed-file checksum must match its
   preprocessing manifest before training. Registry registration is not approval:
   candidates that fail the development baseline remain marked rejected, and
   no deployed alias is assigned by this command.

The one human-readable summary is
[`reports/market_training/market_training_report.md`](reports/market_training/market_training_report.md).
Machine records and restart information stay in `artifacts/market/`; weights
stay in `artifacts/models/`. Progress is saved after every completed symbol.
Repeating the command with identical code/configuration/data/cutoff reuses the
existing verified training result and records the new MLflow run; changed inputs
produce a new candidate. Unexpected training failures are reported, other stocks
continue, and the command exits nonzero. Data exclusions are reported normally.
CSV validation does not verify whether a symbol is an active listed equity or
an index; production onboarding still requires checking the instrument universe.

Superseded processed files/pointers for invalid or missing inputs are preserved
under `artifacts/market/archive/` rather than left as current processed data.
Their previous committed pointers also remain in Git history. A missing raw CSV
keeps its existing raw pointer so you can recover it using DVC if its cached
version is available. It cannot be trained until the raw data is supplied.

For CPU-only single-stock execution, explicitly override the GPU configuration:

```powershell
python -m scripts.train_model --config configs/training.json --device cpu --data-dir data/processed --symbols BIAT
```

The existing GitHub training workflow now explicitly selects CPU. It remains
the single-stock routine workflow, not the market-wide local GPU run.

### DVC upload later

Install `dvc[azure]` when preparing a new environment. The current DVC remote is
already named `azure-blob`, points to `azure://datasets`, and references storage
account `stockspricesdatastore`. Verify that this is the intended account and
subscription before uploading. The `datasets` Blob container must exist and your
identity needs Blob data access. Keep credentials out of Git.

```powershell
az login
python -m dvc remote list
python -m dvc status
# After resolving missing downloads and verifying the destination:
python -m dvc push
```

DVC's Azure remote stores content-addressed cache objects, not a browseable copy
of `data/raw` and `data/processed`. Another machine obtains the chosen Git
revision's `.dvc` pointers and runs `dvc pull` to reconstruct those paths. Do not
point Azure training at an imagined `datasets/data/processed/BIAT.csv` Blob.
Materialize the DVC revision before training, or deliberately publish a separate
Azure ML data asset. [DVC Azure remote authentication](https://github.com/treeverse/dvc.org/blob/main/content/docs/user-guide/data-management/remote-storage/azure-blob-storage.md)

Commit the code/configuration, report, dataset `.dvc` pointers and generated
`.gitignore` changes yourself. Do not commit dataset files, model weights, the
DVC cache, `mlflow.db` or credentials. Model bundles are already recorded in
MLflow; this stage uses DVC for the raw and processed datasets.

## Validation and remaining work

For the student-budget Azure setup, service choices and deployment prerequisites,
follow [Azure setup for the stock forecasting project](AZURE_STUDENT_SETUP.md).

```powershell
python -m pytest tests/forecasting -q
python -m ruff check stock_forecasting tests/forecasting
```

Provider prices do not have verified corporate-action adjustments or exchange
calendars. Horizons count observed bars. Strategy plots include simplified costs
and execution assumptions and do not establish trading profitability. Previously
inspected historical test periods are retrospective evidence.

Live Azure deployment, durable cloud monitoring storage and scheduling remain
subsequent setup work. Prediction monitoring is implemented and tested locally;
real rolling results require issued forecasts followed by fresh outcomes.
A successful benchmark is not proof of a verified live Azure deployment.
The `docs/` directory contains local
project guides and is intentionally excluded from Git; it is absent from a fresh
clone. The existing research CLI and ETL dependencies remain available.
