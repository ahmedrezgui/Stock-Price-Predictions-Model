# Implementation progress

## Stage 1 — audit and corrective fixes

- [x] Read mission, inventory code, workflows, notebooks, tests and local data.
- [x] Preserve pre-existing working-tree changes; no commits mixing user work.
- [x] Architecture audit written before new implementation.
- [x] Baseline focused tests: 48 passed, 6 failed (see architecture audit).
- [x] Corrective fixes and cloud gating verified. LSTM/GRU input validation,
  legacy recurrent shape, frozen fitted scalers, stored sequence lengths, target
  column rejection and GRU CPU serialization corrected. Existing utility tests
  now request their intended 60-row window explicitly and remove duplicate
  arguments. Legacy processing keeps observed rows and rebuilds from raw to
  remove previously fabricated dates. No raw datasets were overwritten.
- [x] Azure train/CD are manual and use `azure-approved`; the legacy deployment
  script has no import-time cloud side effects. No workflows were dispatched.

## Remaining stages

- [x] Stage 2: `config.py`, `data.py`, `storage.py`, config JSON. Strict CSV
  normalization, adjustment policies, calendar validation, causal feature groups,
  publication-time joining, 1/5/20 labels, historical windows, purging, train-only
  scaling, immutable Parquet/checksums and local/Blob adapters. Tests cover
  future mutation invariance, lags, target tails, gaps, invalid data, versioning,
  traversal, calendar target dates, SDK mocks and failure propagation.
- [x] Stage 3: `models.py`, `artifacts.py`. Configurable native recurrent models,
  regression/classification/multitask, normalized target loss, clipping, stopping,
  scheduling, state-dict bundles; train-AIC ARIMA with frozen-coefficient state
  updates and trusted checksum-verified serialization; four naive comparators.
  Zero return is also previous-close price persistence. Tests verify both heads,
  both architectures, CPU restoration and ARIMA future invariance/round trips.
- [x] Stage 4: `evaluation.py`, `backtesting.py`, `statistics.py`, `benchmark.py`,
  `reporting.py`, `summary.py`. Separate regression/direction/probability/IC scores,
  HAC DM, bootstrap intervals, regime scores, training diagnostics, nonoverlapping
  next-open economic simulation, local lineage and optional MLflow. Zero-volume
  execution spans fail rather than selecting trades using future fill knowledge.
  Implemented reports and all 20 requested plot categories, including regime and
  hyperparameter comparisons. Synthetic fixtures are used only for correctness.
- [x] Stage 5: `api.py`, `/v1/predict` and alias, liveness/readiness/catalog,
  typed validation, exact version selection, immutable cache, same preprocessing,
  structured errors/tracing/timing, API keys, null unsupported calendar/probability
  values. Live localhost smoke passed; response saved in `reports/api_smoke.json`.
  Legacy web contracts and LangGraph source were preserved.
- [x] Stage 6: SDK v2 train/register/download/deploy module, scoring adapter,
  environment, Blob/Container Apps Bicep, nonroot Docker targets and five Azure
  guides. Official Microsoft student, quota, model/data-asset, Blob, endpoint and
  pricing/billing documentation reviewed on 2026-10-09. SDK job construction and
  Blob calls tested with mocks; no authentication/provisioning/job/deployment run.
- [x] Stage 7: README, audit, methodology, model comparison/card, API, portfolio
  summary, reports and local verification. No commits were made because the
  working tree contained substantial pre-existing user changes.

## Verified experiments

BIAT raw ilboursa subset, 2018-01-02 to 2026-10-07, 2,187 observations;
source/canonical checksums and dataset version `e2cdd1b2408a383b4dcb` in reports.
All runs use seeds 7/19/43, maximum 12 epochs, equal LSTM/GRU candidate budgets,
validation-selected MSE/Huber, hidden=32/layers=1/sequence=20, price+volatility.

| Run | Final model/baseline rows | Development folds | Tasks |
|---|---:|---:|---|
| `reports/biat-h1` | 23 | 2 per family/task | return, direction, multitask |
| `reports/biat-h5` | 17 | 1 per family/task | return, direction |
| `reports/biat-h20` | 17 | 1 per family/task | return, direction |

42 trained recurrent bundles are local under ignored `artifacts/models`.
57 final benchmark rows, 228 per-run PNG figures plus consolidated charts and
60 paired comparisons are available. ARIMA selected (1,0,1) for the final train
window. Primary conclusions and exact split dates are in model_comparison.md and
reports/evaluation_report.md; no consistent neural architecture winner or trading
edge is established. Training ran on the existing local CUDA-capable environment.
The live API returned health/ready and a real BIAT prediction as of 2026-10-07;
calendar date/probability were correctly null for the unverified return model.

## Verification and failures resolved

- Initial focused legacy run: **48 passed, 6 failed** (recorded in audit).
- New initial unit/integration suite: **24 passed**.
- Broader interim suite: **90 passed**.
- Expanded end-to-end suite: fixed duplicate pytest module names, typed Azure
  job timeout entity and explicit UTF-8 report reading on Windows.
- GPU-enabled combined suite: **104 test assertions passed**, but Python crashed
  during shutdown with native Windows exit `-1073740791` (`0xC0000409`). This is
  **not a clean process-level pass**. The crash reproduces in the legacy eight-test
  PyTorch module alone; disabling CUDA produces six passed/two skipped and exit 0.
  Its native CUDA shutdown fault remains unresolved; no driver/library change was
  attempted. New CUDA benchmark commands and the standalone forecast tests exit
  normally. Machine-readable assertion outcomes are `reports/test_results.xml`.
- CPU compatibility run uses `CUDA_VISIBLE_DEVICES=-1` (an empty value did not
  hide CUDA on this Windows installation), skipping the two legacy CUDA tests.
  **102 passed, 2 skipped; process exit 0.**
  The command is recorded below; results are `reports/cpu_test_results.xml` and
  `reports/test_coverage.json`. Coverage is **91% rounded of the new package**;
  cloud mutation and optional MLflow live paths are not exercised against services.
- Standalone forecasting suite separately verified: **38 passed**, process exit 0.
- Ruff check/format and Python compile checks pass for the new package, new tests
  and Azure scoring script. Figures were generated; benchmark and consolidated
  charts were visually inspected. A Windows text-encoding plot defect was fixed.
- Attempted separate `tests/web/back/test_main.py`: **collection blocked** by
  existing import-time `ChatGoogleGenerativeAI(model=None)` from unconfigured
  Part B. No claim that the whole repository/agent suite passes.
- Docker CLI available, **Linux engine unavailable** (missing Docker Desktop
  named pipe). No container build/runtime verification claimed.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
python -m pytest tests/forecasting tests/prediction_model tests/utils tests/services tests/scripts/test_process_data.py tests/scripts/test_download_data.py tests/scripts/test_download_data_playwright.py tests/scripts/test_train_model.py -q --disable-warnings --cov=stock_forecasting --cov-report=term-missing --cov-report=json:reports/test_coverage.json --junitxml=reports/cpu_test_results.xml
python -m ruff check stock_forecasting tests/forecasting infrastructure/azure/score.py
python -m ruff format --check stock_forecasting tests/forecasting infrastructure/azure/score.py
```

## Explicitly unexecuted / limited

- Azure account access, quota/region/SKU verification, Bicep compilation against
  Azure tooling, chargeable resources, registry uploads, cloud jobs, cloud MLflow,
  deployments, traffic changes and cleanup: **not executed**; require environment
  access and explicit approval where applicable. Managed identity/RBAC not tested
  against an actual account. Docker execution blocked as above.
- Verified split/dividend-adjusted total returns and Tunis trading-session
  targets: **blocked on authoritative provider/calendar evidence**. Observed-bar
  experiments remain usable as honestly labeled engineering research.
- Broader feature ablations, multi-stock cross-sectional IC, formal structural
  breaks, fitted probability calibration/distributional intervals, peak runtime
  memory, prospective evaluation and production drift alerts: **not completed**.
  Feature groups and publication-time context interfaces are available; external
  context datasets were not ingested. Shorting is research-only.
- Only recurrent bundles are exposed by FastAPI; ARIMA is available in training/
  benchmarks and trusted local serialization, not symbol-based online serving.
- Existing independent legacy price/scaler registry path is retained for old
  clients but not the supported return-forecast deployment path. Legacy agent
  startup still needs its own configuration; no LangGraph work was performed.
- Windows native shutdown crash in legacy CUDA tests is unresolved. Use the
  clean CPU compatibility path and standalone forecast suite; do not describe
  the GPU-enabled combined run as a fully passing process merely because its
  test assertions finished successfully.

Cloud provisioning, live deployment and resource deletion require explicit user
approval. No Azure account or resources have been inspected. Corporate-action
adjustments and an authoritative Tunis exchange session calendar are not present
in the supplied data; verified total-return/session-horizon claims are blocked.


## Integration implementation (2026-10-09)

See `integration_guide.md` for the current workflow and remaining operational/product work. Configured notebook, benchmark, CLI and artifact reload now instantiate public LSTMModel/GRUModel classes sharing one recurrent trainer. Legacy constructors remain price-model compatibility adapters. The explanatory notebook was executed successfully in a Jupyter kernel; its original content is preserved separately.

Routine training now produces evaluated candidates, gates local promotion on a training-only baseline comparison, supports explicit fixed-budget production refit, records complete bundles, and skips unchanged normalized data/configuration/implementation. Added active-version selection, rollback, promoted-only release packaging, shared inference in the existing backend, lazy agent initialization, ETL-to-training revision handoff, and a manual forecasting release workflow. No cloud actions were executed.

Final targeted CPU validation: **90 passed, 2 CUDA-only skipped**, exit 0; Ruff, compile and diff checks passed. Notebook execution exited 0. Real BIAT two-epoch smoke training exited 0 with a **rejected** candidate (validation MSE 0.00011099036 versus baseline 0.00010876645), preserving the active registry. The main research benchmark was not rerun. Live GitHub/Azure/container verification, legacy frontend response migration, and persisted realized-forecast monitoring remain outstanding.
