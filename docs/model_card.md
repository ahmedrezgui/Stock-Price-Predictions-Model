# Model card

**Purpose:** educational/portfolio single-stock future-log-return and direction
forecasting. The evaluated stock is BIAT; the code accepts configurable symbols.
It is not a recommendation engine, autonomous trader or calibrated risk service.

**Model families:** native PyTorch LSTM/GRU, shared recurrent encoder with return
and direction heads; single-task objectives and experimental normalized multitask.
Statsmodels ARIMA and train-derived naive baselines benchmark the same origins.
API serves recurrent bundles only; ARIMA is a research benchmark/CLI artifact.

**Training data:** supplied local ilboursa BIAT daily bars, 2018–2026 subset;
2,187 observations. Raw source hash, canonical Parquet hash and dataset version
are embedded in each bundle. No new paid provider was used. No independent
source audit, data redistribution clearance, adjusted-close methodology,
point-in-time corporate-action ledger or official Tunis calendar was available.
Current results are unverified-price returns over observed bars.

**Preprocessing:** no forward fill/interpolation; strictly historical features;
training-input-only StandardScaler; training-label-only target normalization;
explicit feature order/version; purged chronological boundaries. Features are
price/return and volatility by default. Volume/technical groups are optional.
Publication-time as-of joining exists for optional external context, but no macro
or sector context was ingested or evaluated in these experiments.

**Validation:** three seeds, validation-only loss/early-stop selection, two
development walk-forward folds (h=1) or one (h=5/20), final chronological holdout.
See model_comparison.md for verified results. Synthetic fixtures test software
behavior only and never contribute portfolio performance numbers.

**Artifact lifecycle:** a version directory contains CPU-portable state_dict,
metadata including preprocessing/lineage/config/history, and SHA-256 checksums.
Load verifies compatibility and uses torch weights_only. Checksums detect damage,
not malicious replacement of both artifact and hash; accept trusted artifacts
only. Registry must store the whole bundle. Retain previous versions for rollback.

**Limitations:** one stock, fixed small search budget, retrospective data revisions,
unknown corporate actions, sparse sessions, no authoritative session date forecast,
no return intervals, no fitted probability calibration, no portfolio risk model,
no cross-sectional IC, no verified peak memory profiling, no production drift
alerts, no external/live cloud validation. Models were trained before the final
holdout, not refitted on the full history; API smoke testing establishes mechanics,
not freshness of learned parameters or production model promotion.

**Serving:** after completed daily observation, returns/logit head only;
uncalibrated probabilities are labeled; unsupported values null. Strict version
selection, finite-history checks, freshness checks and API-key authentication
for nonlocal mode. Model catalog and OpenAPI support a future LangGraph client
without implementing the agent. No Azure resources were deployed.

**Monitoring plan:** record request latency/error rate, readiness, data age,
feature finite/range failures; later join predictions to matured labels and
compare rolling error/class balance/calibration against training and validation.
Review drift before retraining; do not automatically promote a model based on
one recent good test interval. Live drift alerts are not implemented.
