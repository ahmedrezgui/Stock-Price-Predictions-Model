# Immutable datasets and Azure Blob Storage

Local mode is the default: `FORECAST_STORAGE=local`,
`FORECAST_STORAGE_ROOT=artifacts/storage`. The same `put/get` interface supports
Blob. Parquet snapshots have content-derived identifiers, a SHA-256 checksum,
source-file checksum (benchmark path), schema/provenance metadata and a manifest.
Identical writes are idempotent; different bytes at an existing key fail. Readers
verify checksums. Neither adapter creates containers automatically.

```powershell
python -m stock_forecasting.cli snapshot --data data/raw/BIAT.csv --output reports/snapshot
```

Logical keys: `market-data/processed/{version}/{symbol}.parquet` and manifest.
Use the same immutable adapter for `market-data/raw/{provider}/{symbol}/{version}`,
`market-data/features/{feature_version}/{dataset_version}`,
`experiments/reports/{experiment_id}` and `artifacts/models/{model_version}`.
Benchmarks persist local snapshots deliberately; syncing them to Azure is an
explicit separate action. The model bundle embeds the exact dataset manifest.

After resource approval, review and apply the parameterized Blob Bicep:

```powershell
$storageName = '<globally-unique-lowercase-name>'
az deployment group what-if --resource-group $env:AZURE_RESOURCE_GROUP --template-file infrastructure/azure/storage.bicep --parameters storageName=$storageName
az deployment group create --resource-group $env:AZURE_RESOURCE_GROUP --template-file infrastructure/azure/storage.bicep --parameters storageName=$storageName
$storageId = az storage account show -g $env:AZURE_RESOURCE_GROUP -n $storageName --query id -o tsv
$principalId = az ad signed-in-user show --query id -o tsv
az role assignment create --assignee-object-id $principalId --assignee-principal-type User --role 'Storage Blob Data Contributor' --scope $storageId
$env:FORECAST_STORAGE = 'azure'
$env:AZURE_STORAGE_ACCOUNT_URL = "https://$storageName.blob.core.windows.net"
$env:AZURE_STORAGE_CONTAINER = 'forecasting'
python -m stock_forecasting.cli snapshot --data data/raw/BIAT.csv --output reports/azure-snapshot
az storage blob list --account-name $storageName --container-name forecasting --auth-mode login --query '[].name'
```

Reader-only workloads need Storage Blob Data Reader; ingestion needs Contributor.
Grant roles at the container scope where operationally practical. Managed identity
uses its principal ID and ServicePrincipal type; allow RBAC propagation time.
Microsoft recommends Entra credentials; the SDK implementation uses
DefaultAzureCredential and `overwrite=False`.
[Python Blob authentication and uploads](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-upload-python)

Individual transfer commands (downloads refuse to overwrite local files):

```powershell
python -m stock_forecasting.cli storage upload --file reports/biat-h1/benchmark_summary.json --key experiments/reports/biat-h1/benchmark_summary.json
python -m stock_forecasting.cli storage download --file downloaded-summary.json --key experiments/reports/biat-h1/benchmark_summary.json
```

Bicep disables shared keys and public blobs, enforces TLS, enables versioning and
seven-day soft deletion. Application-level immutability is not regulatory WORM.
Blob versions and soft-deleted data continue occupying storage. Start with Hot LRS
for small experiments; estimate retained GB, transactions and egress. Consider
cooling old raw snapshots only after reviewing retrieval charges. Do not apply an
automatic delete policy to manifests or model versions still referenced by a
deployment. Document a retention decision, then add a reviewed lifecycle rule.
Local fallback requires setting FORECAST_STORAGE back to local explicitly; a
cloud failure is never silently redirected to an untracked local write.
