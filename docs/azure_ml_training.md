# Azure ML training and registry (SDK v2)

The implemented command uses `azure-ai-ml`, `DefaultAzureCredential`, immutable
data assets, a bounded command job, and a versioned custom-model bundle. It does
not start a job on import. The `.amlignore` excludes credentials, local data,
models and reports from uploaded source. Do not put credentials elsewhere in
the source tree. The environment uses Python 3.11/PyTorch and CPU by default.

After approval, create a CPU cluster with minimum zero nodes. First verify the
chosen SKU against your region and subscription quota; the variable is deliberate.

```powershell
$cpuSku = '<approved-available-CPU-SKU>'
az ml compute create --name forecast-cpu --type amlcompute --size $cpuSku --min-instances 0 --max-instances 1 --idle-time-before-scale-down 120
az ml compute show --name forecast-cpu
az ml data create --name biat-raw --version 1 --type uri_file --path data/raw/BIAT.csv
az ml data show --name biat-raw --version 1
python -m stock_forecasting.azure train --dataset azureml:biat-raw:1 --compute forecast-cpu --config configs/forecast.json --approve-cloud-action
```

The output is a job name. Inspect it with `az ml job show --name <job>` and
`az ml job stream --name <job>`. Jobs have a one-hour termination limit. Change
family, task, horizon, seed and feature group in a copied JSON configuration,
and version that configuration before submitting. GPU is optional and requires
separate eligible quota; CPU is adequate for this project's small networks.
CLI `train` now uses family/task from config unless explicitly overridden.

Data assets can also reference an identity-accessible Blob datastore path. Use
`azureml://datastores/<name>/paths/market-data/...` and grant the job identity
read access. Do not assume data asset registration duplicates/versions mutable
external bytes: use the content-versioned keys from this project.
[Data assets](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets?view=azureml-api-2)

The local benchmark always records config, seeds, Git commit and dirty status,
library versions, split dates, feature/dataset versions, histories, metrics, plots
and bundles. Optional MLflow:

```powershell
python -m pip install '.[azure]'
python -m stock_forecasting.cli benchmark --data data/raw/BIAT.csv --output reports/mlflow-run --mlflow-uri sqlite:///mlflow.db
```

For workspace tracking, obtain its tracking URI with
`az ml workspace show --query mlflow_tracking_uri -o tsv`, set
`MLFLOW_TRACKING_URI` and use that value as --mlflow-uri for benchmarks. Azure job
single-model training logs the bundle/history when MLFLOW_TRACKING_URI is set.
Local JSON tracking remains the reproducibility source if MLflow is not installed;
explicitly requested MLflow failures are not silently ignored.

Download job outputs, validate and register a chosen bundle:

```powershell
az ml job download --name '<job-name>' --download-path artifacts/job-output
python -m stock_forecasting.azure register --bundle '<path-to-exact-bundle>' --name biat-return-lstm --approve-cloud-action
az ml model list --name biat-return-lstm -o table
python -m stock_forecasting.azure download --name biat-return-lstm --version '<exact-version>' --output artifacts/registered
```

Model registration preserves weights + preprocessing + feature definition +
dataset lineage together; never independently register the scaler. Compare
validation/holdout reports across immutable versions, keep a model card, and
record the chosen version. Retraining creates a new bundle, not incremental
weight updates under a refitted scaler. The SDK v2 supports local/job/datastore
registration paths. [Model registry](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models?view=azureml-api-2)

No cloud job, asset registration, MLflow service connection or registry download
was executed in this session. CPU/GPU cloud performance remains unmeasured.
