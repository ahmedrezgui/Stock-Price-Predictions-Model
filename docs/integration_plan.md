# Forecasting integration plan

Reviewed local repository code on 2026-10-09. The original plan below is preserved. See `integration_guide.md` for implemented changes, validation and remaining work. GitHub run history, repository environment protections, and deployed Azure resources were not verified.

## Current automation

`.github/workflows/train.yml` automates the legacy training steps after a manual `workflow_dispatch`. It runs `python -m scripts.train_model` on the GitHub runner; the message “Training model on Azure ML” does not correspond to submitting an Azure ML compute job. Azure authentication supports storage and, depending on `MODEL_LOCATION`, model registration.

The earlier implementation changed training and deployment triggers to manual and added the `azure-approved` environment. Merely declaring an environment does not establish that required reviewers are configured in GitHub. The ETL workflow is also manual and does not explicitly invoke training. Consequently, the local workflow definitions do not provide automatic data-update-to-training-to-deployment execution.

The new forecasting CLI, benchmark runner, artifact bundles, and API are separate from that legacy chain. `forecast-ci.yml` runs forecasting tests; it does not train production models or deploy them.

## Findings

| Area | Evidence in code | Consequence |
| --- | --- | --- |
| Model implementation | Notebook and factory use `src/prediction_model/models/LSTMModel.py` and `GRUModel.py`; new benchmark uses `stock_forecasting/models.py` | Two implementations can diverge. |
| Notebook | `notebook/Models benchmark/Models_benchmark.ipynb` calls legacy `train` and `evaluate` | New splits, metrics, artifacts, and reports are not exposed in the user's notebook workflow. |
| Training | `scripts/train_model.py` evaluates an existing model with MAPE, then trains and saves regardless of threshold | The threshold is an alert, not a candidate promotion gate. No new-candidate validation or benchmark report is produced. |
| Artifact identity | Local handlers name files by stock only; `get_or_create_model` loads without checking requested family | Switching model family can silently reuse another family. |
| Azure loading | Both handlers request version 1 and catch broad loading failures | Newly registered versions are not necessarily served; loading errors can be mistaken for absent models. |
| Artifact consistency | Model and scaler are registered separately from the same output directory; model registration precedes the new scaler save | A registered model directory can contain a missing or stale scaler. |
| Serving | `src/services/stock_data_service.py` loads legacy artifacts, requires training through the latest date, and generates calendar-day forecast dates | New bundles are unused; stale-model policy and session-date semantics need explicit handling. Forecasts are also appended as synthetic OHLCV rows. |
| Deployment | `cd.yml` builds `src/web/back/Dockerfile` and `src/web/front/Dockerfile` | It does not deploy the standalone forecasting API built by the root Dockerfile. |
| Configuration and imports | Workflows include obsolete Keras and unrelated agent variables; `scripts/__init__.py` eagerly imports ETL modules | Forecasting execution has avoidable dependency and configuration coupling. |
| Factory error | `src/factories/model_factory.py` returns a `NotImplementedError` object for unsupported names | Invalid configuration fails later rather than at model creation. |
| Production training | New CLI training uses chronological research splits | A distinct production refit policy is needed; benchmark artifacts must not be presented as models fitted through the latest eligible labels. |

## Ordered changes

### 1. Consolidate model construction and training

Preserve the existing public `LSTMModel`, `GRUModel`, and factory entry points. Refactor recurrent internals into one shared implementation with explicit task, feature, window, and horizon configuration. Keep compatibility adapters for legacy close-price behavior; never interpret a legacy price checkpoint as a return or direction model. Raise invalid-family errors immediately.

Acceptance: existing public imports work; notebook and training entry point instantiate the same implementation; serialization round trips preserve predictions, configuration, and feature order. Legacy compatibility tests cover supported old contracts.

### 2. Make the notebook the explanatory benchmark interface

Update the existing notebook with sequential sections: raw data and quality checks; causal features and target examples; chronological splits and purging; train-only scaling; sequence shapes; ARIMA/LSTM/GRU construction; training curves; baseline comparisons; prediction plots; metrics and cost-aware backtests; artifact inspection.

Explain each model's layers, recurrent state, gates, output heads, loss, and hyperparameters beside its construction. Explain each displayed metric's formula, units, preferred direction, and limitations. Include MAE/RMSE, legacy price MAPE where applicable, direction accuracy, precision/recall/F1, ROC-AUC where defined, Brier score/calibration, correlations, and backtest drawdown/returns/turnover/cost assumptions. Expose intermediate tables and arrays by calling reusable functions rather than hiding the entire experiment behind one CLI cell.

Acceptance: a small offline notebook execution succeeds; notebook and CLI produce matching metrics from identical predictions and configuration; cloud actions are not executed by notebook setup cells.

### 3. Integrate `scripts/train_model.py` with the shared pipeline

Retain this familiar workflow entry point and make it a configurable orchestrator around the shared training service. Add validated configuration, symbol selection, explicit local/Azure execution backend, dataset identity, reproducible seeds, and structured per-symbol results. Skip only when both data and effective configuration are unchanged. Handle insufficient history and missing labels explicitly.

Separate research benchmarking from routine retraining. Use development data to select configuration and training duration; evaluate candidates on a time-appropriate validation window. Preserve the historical holdout as a research evaluation, not a repeatedly tuned promotion set. Define a production refit using eligible matured labels and record its training cutoff separately from benchmark cutoffs. Fit preprocessing consistently with the selected refit policy. Monitor deployed forecasts only when their labels mature.

Acceptance: a local fixture produces a versioned bundle and evaluation report; unchanged reruns are skipped; changed family/horizon/configuration creates a new candidate; failures cannot overwrite the active model. Tests mock email and cloud clients.

### 4. Unify artifact loading and promotion

Use one immutable bundle containing weights, scaler, feature order, target semantics, configuration, training cutoffs, dataset identity, and checksums. Identify models by symbol, family, task, horizon, and version. Replace separate stock-only model/scaler lookups with an explicit promoted-version manifest. Do not silently train a replacement when registry access fails.

Define candidate eligibility using validation data, baseline comparisons, integrity checks, and task-appropriate metrics. Keep the active version if a candidate fails. Store promotion history and make rollback select a prior complete bundle.

Acceptance: family mismatch, corrupt bundles, missing scalers, and unavailable versions fail clearly; rejected candidates leave active predictions unchanged; rollback restores the prior version.

### 5. Connect ETL, training, and CI

Make the workflow dependency explicit: validated ETL output and its exact data revision -> tests -> candidate training -> evaluation report and bundle. Do not rely on an incidental bot commit to trigger the next stage. Keep manual execution with explicit inputs, then enable a data-update or scheduled trigger once the local chain passes. Use changed dataset/configuration identity to avoid unnecessary training.

Make execution location truthful: runner execution for the initial CPU path; an explicit Azure ML job submission for the Azure backend. Add timeouts, concurrency control, retained run reports, and minimal job-specific configuration. Keep cloud registration and deployment as separate approved release actions under the original project constraint. Verify actual GitHub environment protections before relying on them.

Acceptance: workflow smoke validation proves the selected revision reaches training; no-data-change runs skip; failed ETL/tests block training; candidate failure blocks promotion. Core CI requires neither agent credentials nor live cloud services.

### 6. Integrate serving and deployment

Route existing backend forecasting through the shared bundle loader and inference service. Preserve legacy response compatibility deliberately, with explicit conversion only when mathematically supported; expose task/horizon/version metadata in the versioned contract. Keep observed history distinct from forecasts. Use verified trading sessions for dates, or report the absence of a verified session calendar without inventing dates. Replace the latest-training-date equality rule with explicit model cutoff and freshness policy.

Align the deployed backend image and startup command with the chosen shared service. Keep agent behavior outside this refactor; isolate initialization boundaries only as needed so forecasting can start and be tested independently.

Acceptance: an offline raw-data -> features -> training -> bundle -> existing API integration test passes; API and notebook predictions agree for the same model/input; container startup and readiness are verified before any release. Actual Azure deployment remains a separate action.

## Completion criteria and outstanding verification

One shared model implementation is used by notebook, CLI, training script, and serving. One artifact contract connects training to inference. Research evaluation, production refit, and model promotion have distinct documented roles. The notebook explains the complete benchmark with inspectable intermediate results. Automation records exact data/configuration/model versions and does not promote failed candidates.

This review did not execute training, send email, dispatch workflows, or deploy resources. Previous test results are recorded in `implementation_progress.md`; they are not fresh verification of this proposed integration. Container runtime, live GitHub configuration, and Azure execution still require verification. Official trading-session and price-adjustment evidence also remains necessary before stronger market-performance claims.
