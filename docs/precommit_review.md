# Pre-commit review — 2026-10-10

This is a tested development milestone, not a complete monitored production system. No staging, commits, application-code changes, live downloads, or cloud actions were performed in this audit.

## Current status

| Requirement | What exists | What is missing |
| --- | --- | --- |
| ilboursa downloads | `scripts/download_data_playwright.py`: multiple symbols, date chunks, form submission, CSV validation, duplicate-row suppression, manual verification fallback | ETL still calls the older requests downloader. Failed Playwright periods are printed without a failing exit status. Provider corrections can create conflicting duplicate dates. Unattended live access is unverified. |
| Preprocessing | `stock_forecasting/data.py`: canonical names, numeric/date parsing, OHLCV validation, duplicates, completed-date filtering, optional calendar checks | `scripts/process_data.py` remains a separate legacy path with French names. It can preserve missing numeric values and catches file errors without failing the command. Need one shared preparation function and quality reports. |
| Models | Configured public LSTMModel/GRUModel classes use a shared PyTorch trainer; ARIMA and naive baselines exist | Legacy constructors retain separate price-model compatibility behavior. A final model family has not been justified across stocks. |
| Notebook | Step-by-step explanations, intermediate data, curves, metrics, backtests and bundle reload | Main demonstration is one stock; the legacy notebook is preserved separately. |
| Multiple-stock benchmark | Per-stock chronological benchmark, seed/loss comparisons and development folds | Persisted substantive results are BIAT h1/h5/h20. No complete multi-stock benchmark command or aggregate model-choice report. Routine training's `--symbols` is not a research benchmark. |
| Prediction API | Locally tested FastAPI, version selection, bundle validation; mounted at `/forecast`; Azure ML scoring adapter exists | Research API still permits family and explicit version selection. Production should serve only the selected deployed model. Local API reads CSVs; Azure ML adapter accepts observations. Reconcile/document these contracts. |
| Blob Storage | Immutable local snapshots/checksums, Blob adapter, DVC workflow, storage instructions | Live integration unverified. Clarify the roles of DVC and direct Blob uploads instead of requiring two competing workflows. |
| Azure ML | Training submission, registration, scoring adapter, environment, Docker target and documentation | No live verification. `forecast-release.yml` currently deploys App Service, not Azure ML. Too many deployment alternatives for the requested simple main path. |
| MLflow | Hooks in benchmark and `stock_forecasting.cli train`; workspace setup instructions | Routine `scripts/train_model.py` / `training.py` do not log metrics/history to MLflow. Azure submission now invokes that script, so the older documentation claiming it logs bundle/history is stale. |
| Degradation gate | Validation versus a simple baseline, local promotion manifest, rollback, bundle/inference checks | No comparison against the deployed model. Beating a baseline does not prove improvement over the incumbent. Release does not require this missing comparison. |
| Deployed accuracy monitoring | Request/error/latency logs and data freshness checks | No persistent forecast ledger, matured-label matching, rolling accuracy report or degradation alert. Runtime health is not prediction quality. |

## Missing-data policy

Handling missing data does not automatically mean filling it. Do not invent weekend/holiday bars or interpolate market prices using future values. Missing OHLCV on an observed session should be reported for re-download or quarantine. A verified exchange calendar is needed to distinguish missing trading sessions from normal closures. Without it, horizons count observed bars. Exclude feature warmup rows explicitly and fit scalers only on appropriate training inputs.

## Simple target design

Keep the current repository, familiar public classes, ordinary Python functions, small configuration files and short scripts. Share one normalization function, trainer, evaluation function and inference function. Let the notebook explain and call those functions.

Use Playwright for downloads, pandas/scikit-learn for preparation, PyTorch for recurrent models, statsmodels for ARIMA, MLflow for experiment records, Blob for versioned data/reports, Azure ML for the chosen deployment path, and GitHub Actions to run scripts in order. FastAPI remains useful locally. App Service and Container Apps need not be additional required forecasting deployment targets. Preserve existing DVC history while deciding its ongoing role.

Initially, selecting LSTM means one chosen model family with separately fitted models per symbol. One pooled model covering all stocks would be a separate experiment; it is not currently implemented.

## Next changes, in order

### 1. Finish the data path

Make `scripts/process_data.py` call strict normalization and write canonical prepared files plus per-symbol quality reports. Keep compatibility exports only where necessary. Wire ETL to Playwright with explicit browser setup. Record failed ranges and return failure if required inputs are incomplete. Continue downloading other symbols if useful, but block dependent training. Reconcile provider corrections by symbol/date deliberately. Retain local manual verification fallback without promising unattended access when the provider blocks it.

Acceptance: download fixture -> canonical data -> training input works; repeated downloads are idempotent; incomplete or invalid required data blocks the workflow.

### 2. Produce model-choice evidence

Add a small `benchmark_models.py` loop around the existing benchmark function. Use a reviewed basket of stocks, adequate history, matching evaluation dates, seeds and training budgets. Compare naive baselines, ARIMA, LSTM and GRU. Report per-stock metrics/sample counts and aggregate normalized errors relative to baseline plus mean/median ranks. Record exclusions and failures; do not average raw price errors across stocks.

Select on development validation. Report the frozen choice on an untouched final period. Previously inspected BIAT holdouts must not be called new unseen evidence. Save one report covering accuracy, robustness, training/inference time and limitations. LSTM remains a candidate until the results justify it; no convincing neural-model advantage is also a valid result.

### 3. Connect MLflow consistently

Use one small logging helper from research and routine training. Record symbol, family, horizon, seed, data/code versions, split cutoffs, losses, validation metrics, selected epoch and artifact version. Verify locally first, then against the Azure ML workspace. Keep JSON reports readable independently and correct stale documentation.

Azure ML can provide the tracking backend without a separate hosted tracking server: [Microsoft MLflow configuration guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-configure-tracking?view=azureml-api-2).

### 4. Enforce the deployment decision

Evaluate the candidate and deployed model on exactly the same eligible observations, each with its frozen scaler. Both models' training and selection cutoffs must precede that evaluation window. Define a primary metric, minimum sample size and allowed regression in configuration. Start with return RMSE/MAE; report direction accuracy as supporting evidence and require baseline checks. Retain the incumbent when evidence is insufficient or a check fails.

Write a decision JSON containing exact model/data versions, dates, scores, thresholds and reasons. Deployment must verify it. A successful image build or registry entry is insufficient. A subsequent refit changes the model version: the old candidate decision does not independently validate the new weights. Decide whether to deploy the tested artifact or wait for evidence for the refitted version.

Use a distinct Azure ML deployment version without changing live traffic, smoke-test it, then switch traffic after the gate passes. Keep the previous version for rollback; avoid overwriting a fixed `candidate` deployment. [Microsoft safe rollout guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-safely-rollout-online-endpoints?view=azureml-api-2).

### 5. Add understandable monitoring

Persist forecasts with symbol, origin, horizon, prediction, model version and data version. A scheduled Python script joins only matured outcomes after new closes arrive, then calculates rolling RMSE/MAE and direction accuracy with sample counts. Pending labels stay pending. Save reports in Blob/MLflow and expose failed checks in GitHub Actions; messaging integrations are unnecessary initially.

For a 20-observation forecast, wait for the twentieth future observation before scoring. Separate metrics by model version. Compare with declared reference windows and thresholds. This detects measured deterioration; it cannot guarantee future performance.

## Commit readiness

The tree can be committed as a reviewed development checkpoint, not as a finished production system. There are pre-existing unrelated changes, including deletions under `src/recommender_system/`. Review those separately. Select intended reports and notebook backups deliberately rather than staging everything.

Possible user-created commit groups: ingestion/preprocessing; models/benchmark/notebook; artifacts/API; workflows/cloud documentation. Keep dependencies coherent; one clearly labeled integration checkpoint is preferable to a broken split. No staging or committing was done here.

## Fresh verification

On 2026-10-10, targeted CPU tests completed with **100 passed, 2 CUDA-only skipped**, exit 0. Coverage includes forecasting, public models, utilities/preprocessing, training entry point, stock service/backend and Playwright. The browser test used a local fixture page, not live ilboursa. Ruff checks passed for the forecasting package and relevant tests/training entry point. This was not a full repository test run or external-service verification. Local log: `reports/precommit_audit_tests.log`.
