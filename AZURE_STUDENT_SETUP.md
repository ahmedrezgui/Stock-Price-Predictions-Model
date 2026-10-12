# Azure setup for the stock forecasting project

Prepared 10 October 2026. This is an operator guide, not a record of resources
already deployed. Commands below create billable resources when you run them.
Prices, student quotas and available regions must be checked in your subscription.

## 1. Recommended services and why

For five stocks, daily CSV updates, a small recurrent model and occasional API
requests, use the following arrangement:

| Need | Choice | Reason |
| --- | --- | --- |
| Raw and processed datasets | Azure Blob Storage, Standard StorageV2, Hot, LRS | File/object storage fits CSV, Parquet and model bundles. No database or data lake processing platform is needed. LRS is an economical choice when you also retain local copies. |
| Experiment tracking | Azure ML workspace, initially without compute | Central MLflow tracking while training runs on your existing local GPU. |
| Production model records | Azure ML workspace model assets, after a registry adapter is implemented and tested | One place for exact model versions and lineage, without operating a separate MLflow server. This is not a drop-in replacement for the current alias-based registry. |
| Prediction API | Azure Container Apps, Consumption, minimum 0 and maximum 1 replica | Runs the existing FastAPI Docker application, provides HTTPS and can scale to zero for low traffic. |
| Training | Your local GPU first | Avoids spending credits on work your current hardware already performs. Use Azure ML CPU jobs only when you specifically need to demonstrate cloud training. |
| Forecast monitoring persistence | Blob records in the same storage account, after a storage adapter is implemented | Small immutable forecast/outcome records fit object storage. Avoid an always-running SQL server and avoid SQLite on network storage. |
| Daily data/monitoring jobs | Local scheduling first; Container Apps scheduled Jobs later | Learn the data flow locally before adding cloud automation and its running costs. |

These are project-specific recommendations, not a claim that every service is
free. Azure ML has no additional platform charge, but underlying compute,
storage, image registry and logging can incur charges. [Azure ML pricing](https://azure.microsoft.com/en-us/pricing/details/machine-learning/)

Container Apps Consumption includes monthly grants of 180,000 vCPU-seconds,
360,000 GiB-seconds and 2 million HTTP requests per subscription. Compute usage
is not charged while scaled to zero. At **1 vCPU and 2 GiB**, the compute grants
correspond to about **50 allocated replica-hours per month**, assuming no other
apps/jobs consume the grants. Startup and scale-down time also consume resources.
This is not 50 hours of model computation and does not cover storage or logs.
[Container Apps billing](https://learn.microsoft.com/en-us/azure/container-apps/billing)

Do not start with AKS, Databricks, a GPU endpoint, a continuously running notebook
VM or a paid App Service plan. They add cost or operations that this workload
does not yet justify. Azure ML managed online endpoints remain a useful optional
learning exercise, but provisioned inference compute is a poor starting point
for occasional requests on a $100 total budget. CPU inference should be enough
for this small model; verify latency and memory before selecting a larger size.

Azure ML can manage experiments and models even when the API runs in Container
Apps. An Azure ML managed endpoint is not required merely to use Azure ML.

## 2. Understand the current repository before provisioning

| Existing component | What works now | What still needs work for the recommended deployment |
| --- | --- | --- |
| `infrastructure/azure/storage.bicep` | Private Blob container, HTTPS, LRS, versioning and seven-day soft delete | Deploy it, then grant data-access roles. |
| `stock_forecasting/storage.py` | Identity-based Blob upload/download and immutable processed snapshots | Serving does not yet refresh its local CSVs from Blob automatically. |
| MLflow tracking | Parameters, metrics and artifacts are implemented | Test against an actual Azure ML workspace using `azureml-mlflow`. |
| `stock_forecasting/model_registry.py` | Standard MLflow custom bundles and candidate/deployed aliases | Adapt custom model registration and deployment-state lookup for Azure ML. |
| `stock_forecasting/release_gate.py` | Same-data comparison and fail-closed release protection | Preserve these checks when changing the registry/deployment adapter. |
| `stock_forecasting/monitoring.py` | Local SQLite, matching and rolling metrics | Add durable remote storage before using ephemeral cloud containers. |
| `infrastructure/azure/container-app.bicep` | Starting template for API, HTTPS, identity and scale-to-zero | Complete storage integration, image authentication, startup behavior and release binding. It is not currently a production-ready deployment template. |
| `.github/workflows/forecast-release.yml` | Gate-aware App Service release path | Change its deployment target to Container Apps; authenticate to Azure before Azure-backed tracking/gate operations. |
| `stock_forecasting/azure.py` | Explicit training, download and custom model registration helpers | Direct managed-endpoint deployment is intentionally disabled. |

**Important compatibility point:** our saved bundles are custom directories, not
MLflow flavor/pyfunc models. Microsoft's capability table distinguishes custom
model registration through Azure ML SDK/CLI from MLflow-format registration.
Do not assume changing `MLFLOW_TRACKING_URI` migrates our whole registry. The
current `candidate` and `deployed` alias calls also need a live compatibility
test or an Azure-specific replacement. [Azure ML registry capabilities](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models-mlflow?view=azureml-api-2)

The production gate currently requires 120 eligible observations after the policy
freeze and model-selection/registration cutoffs. You can set up storage and
tracking immediately. A production release still requires a passing gate; do
not lower thresholds, backdate registration or reuse inspected benchmark dates
just to complete a cloud demo. A configured Azure workspace is not evidence that
a model has passed release evaluation.

## 3. Protect the student credit first

Azure for Students provides $100 for eligible services over 12 months, not $100
every month. Check the subscription expiry and remaining credit in the Azure
Sponsorships portal. Do not switch to pay-as-you-go merely to get a deployment
command to work. [Student subscription and credit](https://learn.microsoft.com/en-us/azure/cost-management-billing/manage/azurestudents-subscription-disabled)

1. In Azure Portal, open **Subscriptions** and select **Azure for Students**.
2. Confirm it is the subscription you will use and retain its spending protection.
3. Open **Cost Management → Budgets** at subscription scope, if available for your
   offer. Create a monthly **$5** planning budget with alerts at 50%, 80% and 100%.
4. Add your own email as the recipient. If budgets are unavailable, check the
   sponsorship credit dashboard manually at least weekly and after each experiment.
5. Keep a simple allocation: up to $60 across the year for storage/tracking/logs
   and light serving, $25 for short experiments, $15 untouched reserve. These are
   spending limits you choose, **not estimated Azure bills**.
6. In Cost Analysis, group by **Resource**, then by **Service name**. Look for
   unexpected VM, Container Registry, Log Analytics and storage costs.

Budget alerts do not stop resources and are not real-time hard caps. Usage can
take time to appear. [Budget behavior](https://learn.microsoft.com/en-us/azure/cost-management-billing/costs/tutorial-acm-create-budgets)

Use the [Azure pricing calculator](https://azure.microsoft.com/en-us/pricing/calculator/)
before deploying. Enter your actual region, a small measured storage amount,
expected Container Apps active hours, log ingestion and any registry costs.
An example calculator scenario is 5 GB Blob data and 10 API replica-hours/month
at 1 vCPU/2 GiB; it is a scenario to price, not a promised monthly bill. Include
versioned blobs, MLflow artifacts, outbound transfer and scheduled jobs.

## 4. Prepare your laptop and choose a region

Use PowerShell from the repository root. Install Azure CLI and Docker Desktop
if needed. Activate the existing Python virtual environment. Install the project's
Azure integration in a separate environment if you want to avoid changing the
already-tested local MLflow dependency set:

```powershell
python -m pip install -e ".[azure]"
python -m pip check
az login
az account list --output table
az account set --subscription "YOUR-STUDENT-SUBSCRIPTION-ID"
az account show --query "{name:name,id:id,tenantId:tenantId}" --output table
az extension add --name ml --upgrade
az extension add --name containerapp --upgrade
```

Start by checking **West Europe**, but use a region allowed by your student
subscription that offers both Azure ML and Container Apps. Availability and
quotas vary; West Europe is a proposed choice, not a guarantee. Put storage,
workspace and serving in the same region where possible. If deployment reports
`RequestDisallowedByAzure` or a quota error, inspect the offer's allowed regions
and quotas instead of upgrading or creating resources in several regions.

Set your own values. Storage names must be globally unique, lowercase letters
and digits, 3–24 characters:

```powershell
$subscriptionId = az account show --query id --output tsv
$location = "westeurope"
$resourceGroup = "rg-stock-forecast-student"
$storageName = "ststockYOURUNIQUESUFFIX"  # Replace with lowercase letters/digits.
$workspaceName = "ml-stock-forecast"
az group create --name $resourceGroup --location $location --tags project=stock-forecast environment=student
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

Provider registration can take several minutes. Store resource names and IDs in
your personal setup notes; do not put access keys or tokens into Git.

## 5. Create the dataset storage account

Use the existing template:

```powershell
az deployment group create --resource-group $resourceGroup --name dataset-storage --template-file infrastructure/azure/storage.bicep --parameters storageName=$storageName containerName=forecasting location=$location
$storageId = az storage account show --resource-group $resourceGroup --name $storageName --query id --output tsv
$containerScope = "$storageId/blobServices/default/containers/forecasting"
$myObjectId = az ad signed-in-user show --query id --output tsv
az role assignment create --assignee-object-id $myObjectId --assignee-principal-type User --role "Storage Blob Data Contributor" --scope $containerScope
```

In Portal verify: Standard StorageV2, LRS, Hot access tier, no anonymous Blob
access, TLS 1.2 minimum, versioning enabled and seven-day soft delete. Leave
hierarchical namespace off; this project does not need ADLS Gen2 features.
The template disables shared-key authentication: use Entra identity, not a
connection string. Network access and anonymous access are different settings;
an authenticated public service endpoint is sufficient for this initial setup.

Role assignment requires appropriate access-management permission. Allow time
for a new assignment to propagate. Subscription/resource Contributor alone does
not provide Blob data access. [Blob data roles](https://learn.microsoft.com/en-us/azure/storage/blobs/assign-azure-role-data-access)

Validate access:

```powershell
az storage blob list --account-name $storageName --container-name forecasting --auth-mode login --output table
```

An empty list is correct for a new container. An authorization failure is not a
reason to enable public access.

## 6. Upload raw data and processed snapshots

Configure the existing storage adapter in this terminal:

```powershell
$env:FORECAST_STORAGE = "azure"
$env:AZURE_STORAGE_ACCOUNT_URL = "https://$storageName.blob.core.windows.net"
$env:AZURE_STORAGE_CONTAINER = "forecasting"
```

Upload the five local CSVs under content-hash paths. A changed file gets a new
path; an unchanged file can be uploaded again without creating a new version:

```powershell
foreach ($symbol in @("BIAT", "ATB", "ARTES", "NAKL", "CIL")) {
    $csvPath = "data/raw/$symbol.csv"
    $fileHash = (Get-FileHash -LiteralPath $csvPath -Algorithm SHA256).Hash.ToLower()
    $blobKey = "market-data/raw/$symbol/$fileHash.csv"
    python -m stock_forecasting.cli storage upload --file $csvPath --key $blobKey
    if ($LASTEXITCODE -ne 0) { throw "Upload failed for $symbol" }
    Write-Output "$symbol : $blobKey"
}
```

Save those paths: a reproducible run references an exact version, not a mutable
`latest.csv`. The existing `snapshot` command performs canonical normalization
and stores a versioned Parquet file plus manifest. For BIAT:

```powershell
python -m stock_forecasting.cli snapshot --data data/raw/BIAT.csv --config configs/training.json --output artifacts/azure-setup/BIAT
```

Inspect `artifacts/azure-setup/BIAT/manifest.json` and verify that its `key` exists
in Blob Storage. The command uses the config's date range: the current
`configs/training.json` now starts in 2010 for market-wide LSTM training.
Use a separate config with the correct symbol when processing other stocks.
Do not silently change the frozen model's training configuration. Repeat with
one matching config per stock; a BIAT config must not normalize ATB data.

Download one uploaded raw file to a **new** local filename and compare its hash:

```powershell
# Set this to the BIAT hash printed/derived during upload.
$biatHash = (Get-FileHash data/raw/BIAT.csv -Algorithm SHA256).Hash.ToLower()
New-Item -ItemType Directory -Force artifacts/azure-setup | Out-Null
python -m stock_forecasting.cli storage download --file artifacts/azure-setup/BIAT-downloaded.csv --key "market-data/raw/BIAT/$biatHash.csv"
Get-FileHash data/raw/BIAT.csv, artifacts/azure-setup/BIAT-downloaded.csv -Algorithm SHA256
```

The hashes must match. Existing download destinations are deliberately not
overwritten. Keep original raw data, model-referenced processed versions and
release evidence. Later, add lifecycle rules for expendable older versions and
logs; do not apply a blanket delete policy to training or release evidence.

## 7. Create Azure ML for tracking, without compute

Create a regular Azure Machine Learning workspace, not a Foundry hub:

```powershell
az ml workspace create --name $workspaceName --resource-group $resourceGroup --location $location
$env:AZURE_SUBSCRIPTION_ID = $subscriptionId
$env:AZURE_RESOURCE_GROUP = $resourceGroup
$env:AZURE_ML_WORKSPACE = $workspaceName
$azureTrackingUri = az ml workspace show --name $workspaceName --resource-group $resourceGroup --query mlflow_tracking_uri --output tsv
```

Review the resources Azure creates alongside the workspace. Keep its default
storage separate from the tightly configured dataset account initially; this
avoids mixing workspace dependency requirements with your data storage policy.
Do not create a compute instance, compute cluster or endpoint yet. Container
Registry may be created later when an operation needs it; monitor that cost.
[Workspace creation and dependencies](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace-cli?view=azureml-api-2)

Open `https://ml.azure.com`, select your subscription/workspace and locate
experiments/jobs and model assets. A workspace is an organizational and tracking
resource; it does not mean a training VM is running.

Test **tracking only** before switching routine training. In this terminal:

```powershell
$env:AZURE_TRACKING_TEST_URI = $azureTrackingUri
@'
import os
from mlflow import MlflowClient

client = MlflowClient(tracking_uri=os.environ["AZURE_TRACKING_TEST_URI"])
name = "azure-connectivity-test"
experiment = client.get_experiment_by_name(name)
experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
run = client.create_run(experiment_id)
client.log_param(run.info.run_id, "purpose", "tracking-connectivity-only")
client.log_metric(run.info.run_id, "connectivity_check", 1.0)
client.set_terminated(run.info.run_id, "FINISHED")
print(run.info.run_id)
'@ | python -
```

Verify that run in Studio. It is not a model-quality metric. Local Azure CLI
login can supply credentials through the Azure ML MLflow plugin.
[Configure remote MLflow tracking](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-configure-tracking?view=azureml-api-2)

Keep existing local `mlflow.db` and `mlruns` intact. Changing an URI does not
copy prior runs, artifacts or registry entries. Do not run the complete routine
training/registration path against Azure until the next section is addressed.

## 8. Make datasets visible in Azure ML

In Azure ML Studio:

1. Open **Data → Datastores → Create**.
2. Name it `stockdata`; choose Azure Blob Storage.
3. Select the dataset storage account and `forecasting` container.
4. Choose identity-based access rather than storing an account key.
5. Grant the workspace managed identity **Storage Blob Data Reader** on this
   container if workspace browsing requires it. Your own user already has data
   access. Future compute/jobs need their execution identity granted access too;
   permissions do not transfer automatically between identities.
6. Create a Data asset named `biat-raw`, type **File / uri_file**.
7. Point it to the exact path
   `azureml://datastores/stockdata/paths/market-data/raw/BIAT/<hash>.csv`.
8. Use a version tied to the content hash and describe its dates/source.
9. Repeat for other symbols if needed. Keep processed Parquet assets separate.

A datastore is a connection/reference to storage. A data asset is a versioned
reference to a file/folder. Neither is preprocessing. Creating from an existing
datastore path avoids unnecessarily re-uploading the same local files.
[Data assets](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets?view=azureml-api-2)

## 9. Complete these code changes before cloud model release

This section is a required implementation plan, **not functionality already
available through a secret environment variable**:

1. **Azure registry adapter:** register verified bundles as Azure ML custom model
   assets via SDK v2. Preserve exact training run ID, artifact path, checksum,
   dataset version and registration time. Test search, download and checksum
   verification. Define explicit candidate/deployed version records compatible
   with Azure; update gate and deployment confirmation to use them. Keep one
   release authority and serialize updates. Do not replace `deployed` with
   “highest version”.
2. **Remote forecast store:** preserve the SQLite store for local work; add Blob
   records keyed by forecast ID and immutable outcome/reconciliation evidence.
   Use conditional creation/updates for duplicate and concurrent requests.
   Reuse existing matching and metrics logic. Do not upload a SQLite database
   after every request: concurrent writers and restarts can lose forecasts.
3. **Data refresh:** fetch the latest explicitly published valid CSV snapshot
   from Blob using managed identity, verify its hash, and replace the serving
   cache atomically. Refresh on cold start and periodically while running.
   Record the exact snapshot used by each forecast. Keep the existing stale-data
   rejection. A release image's baked-in CSV will otherwise become stale.
4. **Container Apps release adapter:** bind the passing gate to the exact image
   digest and model checksum, deploy the approved revision, check HTTPS readiness
   and default prediction version, then update deployed state. Keep rollback
   information for the previous healthy image. On failure, do not update the
   deployed record.
5. **Cloud authentication and packaging:** include the Blob SDK and
   `azure-identity` in the serving image; the current base Dockerfile installs
   only the project's base dependencies. Add managed identity configuration and
   an image-pull credential/identity as required. Test Azure ML dependencies
   separately from serving dependencies.

Replica-local storage is temporary. Setting `FORECAST_MONITOR_DB=/tmp/...` does
not make monitoring durable. A network-mounted SQLite file is not the proposed
solution either. [Container Apps storage lifetimes](https://learn.microsoft.com/en-us/azure/container-apps/storage-mounts)

Acceptance tests: a forecast survives app restart/scale-to-zero; duplicate
requests create one record; a new CSV is used without retraining; outcomes match
after maturity; a failed release leaves deployed state unchanged; model/data
versions in the API, gate, registry and monitoring agree.

## 10. Prepare and approve the model release

Start with **BIAT**, because the existing release policy is scoped to BIAT,
one-observation return prediction. This is an operational starting point, not
a claim that BIAT or LSTM is a proven investment winner.

For the existing local registry, these commands already work:

```powershell
python -m stock_forecasting.release_gate --mlflow-uri sqlite:///mlflow.db --name stock-BIAT-return-h1 --version YOUR-CANDIDATE-VERSION --data data/raw/BIAT.csv --bootstrap
```

`--bootstrap` is only for the first deployment with no incumbent. It still
requires evidence and baseline thresholds. For later candidates, omit it.
Read the resulting gate run and reasons. A blocked result means stop release,
not retry with different evaluation dates.

Only after a pass, package the exact candidate using its gate run ID and a new
output directory:

```powershell
python -m stock_forecasting.release --mlflow-uri sqlite:///mlflow.db --gate-run-id YOUR-PASSING-GATE-RUN-ID --data data/raw --output deployment
docker build --target packaged -t stock-forecast:YOUR-RELEASE-TAG .
```

Cloud releases must use the tested shared Azure registry/tracking adapter from
section 9, rather than a GitHub runner attempting to open your local database.
The current gate expires after 24 hours and is tied to policy, model and data
hashes. Recheck immediately before deployment.

For a first personal experiment you can keep release orchestration on your
laptop, where the local registry exists. Automated GitHub release needs a shared
registry and Azure authentication. Neither path removes the monitoring/data
storage requirements in section 9.

## 11. Deploy the API to Container Apps after section 9 is complete

These are the target Portal settings. Do not deploy the current incomplete
template and work around failed readiness checks by disabling monitoring.

1. Push the approved image under an immutable release tag to your chosen
   container registry. Prefer a private repository if the image includes data
   or weights you do not intend to publish. Use an existing external registry
   initially to avoid an additional Azure Container Registry subscription cost.
   If you choose ACR Basic for Azure-native image access, price it separately and
   grant the app image-pull access through managed identity.
2. In Portal, create **Container App**, resource group `rg-stock-forecast-student`,
   app name `stock-forecast-api`, and an environment in the same region.
3. Select **Consumption** for the workload; do not provision dedicated workload
   capacity. Review the logging resource and ingestion cost before creating it.
4. Select your approved image/tag or digest and configure private-registry access
   through the supported secret or managed-identity mechanism.
5. Allocate **1 vCPU / 2 GiB** initially. Reduce only after measuring peak startup
   memory and prediction latency. Set **minimum replicas 0**, **maximum 1**.
6. Enable HTTP ingress, target port **8000**, external access, HTTPS only.
7. Enable the app's **system-assigned managed identity**.
8. Grant that identity **Storage Blob Data Reader** on the dataset container.
   Create a separate private `forecast-monitoring` container and grant it
   **Storage Blob Data Contributor** there. The forecasting app should not need
   write access to training datasets.
9. Configure `FORECAST_ENV=production` and `FORECAST_API_KEY` as an app secret.
   Set `FORECAST_MAX_AGE_DAYS=14` unless you have deliberately revised the data
   freshness policy. Configure the remote-monitoring and data-refresh settings
   defined by the new adapters. Those environment variable names do not exist
   in the current implementation and should be documented with that change.
10. Set readiness to `/ready` and liveness to `/health`. Give startup sufficient
    time for PyTorch import and data/model loading; measure this in the actual
    image. Use an appropriate startup probe before tightening readiness limits.
11. Copy the generated `https://...azurecontainerapps.io` URL. No custom domain
    or separately purchased certificate is needed for the initial demonstration.
12. Make one authenticated prediction and verify its forecast ID in remote
    monitoring storage. Then restart/scale down and confirm the record remains.

Managed identities provide Azure access without storing storage account keys in
the application. [Container Apps managed identity](https://learn.microsoft.com/en-us/azure/container-apps/managed-identity)

With your endpoint URL and API key supplied in the current terminal:

```powershell
$endpoint = "https://YOUR-APP.azurecontainerapps.io"
Invoke-RestMethod "$endpoint/ready"
$headers = @{ "X-API-Key" = $env:FORECAST_API_KEY }
$body = @{ symbol = "BIAT"; model = "lstm"; horizon = 1; task = "return" } | ConvertTo-Json
Invoke-RestMethod "$endpoint/v1/predict" -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

The default request must serve the approved version. Test an unauthorized
request too: it should return 401. A correct HTTP response is not sufficient;
verify the version, dataset freshness and saved forecast. Only the release
confirmation path should update the deployed model record.

Expect a slower first request after scale-to-zero. Do not add frequent external
keep-alive requests unless you explicitly want to pay for an always-warm API.

## 12. Schedule data refresh and outcome matching

Initially keep the existing downloader/preprocessing on your laptop. After a
successful validated refresh, publish a new immutable raw/processed version and
update the cloud serving manifest through the new adapter. A forecast API does
not automatically download ilboursa data merely because Blob access is configured.

For cloud automation later, create one **scheduled Container Apps Job** in the
same Consumption environment. Run the sequence:

1. Download and validate data; leave the previous valid snapshot available on failure.
2. Publish immutable files and atomically update the serving snapshot reference.
3. Read persisted forecasts, reconcile mature outcomes and calculate rolling errors.
4. Save one current Markdown report to Blob and log monitoring metrics/artifacts
   to the Azure ML workspace. Record failures rather than presenting stale results
   as current. Grant the job identity Blob access and workspace tracking permissions.

Use one parallel execution, a bounded timeout and limited retries. Playwright
requires browser binaries/system dependencies in the job image; the current API
image is not a complete downloader/job image. Verify each command manually before
scheduling it. Example schedule: `30 5 * * 1-5` means weekdays at 05:30 **UTC**;
choose a time when the provider has published the previous completed session.
Holidays and publication delays are handled by data validation, not by assuming
the cron schedule is an exchange calendar. [Scheduled job behavior](https://learn.microsoft.com/en-us/azure/container-apps/jobs)

Use GitHub OIDC for future automated Azure login, with permissions scoped to this
project. The existing workflow's Azure login happens after its gate; an
Azure-backed gate needs login before that operation. A normal MLflow token alone
is not a substitute for the Azure ML plugin and Azure identity configuration.

## 13. If you specifically need an Azure ML managed-endpoint demonstration

Treat it as a short, separately budgeted exercise after the gate-aware adapter
is implemented:

1. Confirm CPU quota and supported endpoint VM SKUs in the selected region.
2. Price the least suitable-cost CPU SKU that actually fits the model and
   environment; do not assume the smallest general Azure VM is supported.
3. Register the checksum-verified custom bundle with Azure ML SDK v2, preserving
   training lineage and gate evidence.
4. Supply the custom scoring script/environment and a durable remote monitoring
   backend. The bundle is not currently suitable for no-code MLflow deployment.
5. Deploy one instance, invoke it, verify outcomes/lineage, and record the demo.
6. Delete the deployment/endpoint immediately afterward if it is no longer needed.
   Closing Studio does not delete inference compute. Do not plan around
   request-based scale-to-zero for a provisioned managed online deployment.

Use the Portal estimate or pricing calculator: compute cost is approximately
`instance hourly price × instance count × running hours`, plus supporting
resources. For illustration only, a $0.10/hour resource running 730 hours would
consume $73 before other charges; $0.10 is not a quoted endpoint SKU price.
[Managed online endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online?view=azureml-api-2)

The project currently rejects direct Azure ML deployment. Implement its release
binding and endpoint-specific verification first; do not remove that guard just
to follow a generic Azure tutorial.

## 14. Weekly checks and cleanup

- Check remaining credit, expiry and costs by resource. Investigate any unexpected
  spend before creating another service.
- Verify API/data freshness, latest monitoring outcome date and failed job runs.
- Check that Container Apps still has min 0/max 1 and no accidental dedicated
  compute. Keep only the logs/artifact copies you need.
- Keep the exact deployed model, previous rollback model, referenced data and
  release evidence. Export irreplaceable artifacts before subscription expiry.
- Remove temporary online endpoints, compute instances, clusters and registries
  after experiments; inspect associated disks/storage/logging separately.
- For total teardown, first back up needed data and check every resource in
  `rg-stock-forecast-student`, then use **Delete resource group** in Portal.
  This destroys the workspace, storage and services in that group.

Practical order: finish sections 3–8 first, then implement/test section 9, then
release an eligible model and deploy sections 10–12. This avoids paying for an
endpoint while the model is still waiting for release evidence.
