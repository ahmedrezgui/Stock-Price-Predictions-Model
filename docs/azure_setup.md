# Azure setup — unexecuted cloud steps

Official Microsoft sources were checked on 2026-10-09. No Azure CLI, authenticated
subscription, region quota or live resource inventory was available in this
session. All commands below are operator instructions; **cloud writes need
explicit approval**. Existing DVC config refers to a datasets container and an
account name, but that is configuration evidence, not proof resources exist.

Use local training and local storage first. Suggested cloud footprint: one private
Blob container, optionally an Azure ML workspace for jobs/registry, and one
scale-to-zero Container Apps consumption service for demonstrations. Reuse a
suitable existing App Service only after checking its current plan/cost. A managed
online endpoint is an optional paid comparison, not the default deployment.

Azure for Students advertises a $100 credit for 12 months, subject to eligibility
and current terms. It does not establish your region/SKU quota or guarantee that
an always-on endpoint is free. Verify the active offer in the portal.
[Student offer](https://azure.microsoft.com/en-us/free/students/)

```powershell
az login
az account show --query '{name:name,id:id,state:state}'
$env:AZURE_SUBSCRIPTION_ID = '<approved-subscription>'
$env:AZURE_RESOURCE_GROUP = '<approved-resource-group>'
$env:AZURE_ML_WORKSPACE = '<approved-workspace>'
$location = '<approved-region>'
az account set --subscription $env:AZURE_SUBSCRIPTION_ID
az extension add --name ml --upgrade
az provider show --namespace Microsoft.MachineLearningServices --query registrationState
az provider show --namespace Microsoft.App --query registrationState
az ml compute list-sizes --location $location -o table
az ml compute list-usage --location $location -o table
```

Check Azure ML workspace/compute availability, VM-family cores, endpoint quotas
and regional capacity in the portal before selecting a CPU SKU. Managed endpoints
can need extra quota for upgrade capacity; the [quota guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2)
explains the reservation. Student quota increases are not guaranteed.

After approval, create only missing resources:

```powershell
az group create --name $env:AZURE_RESOURCE_GROUP --location $location
az ml workspace create --name $env:AZURE_ML_WORKSPACE --resource-group $env:AZURE_RESOURCE_GROUP --location $location
az configure --defaults group=$env:AZURE_RESOURCE_GROUP workspace=$env:AZURE_ML_WORKSPACE
az ml workspace show --name $env:AZURE_ML_WORKSPACE --resource-group $env:AZURE_RESOURCE_GROUP
python -m pip install '.[azure]'
```

Workspace creation has dependent storage/monitoring/key-vault resources; inspect
the resulting billable footprint. SDK v2 authentication uses DefaultAzureCredential:
`az login` locally, managed identity in hosted jobs/apps. Assign only resource-
scoped data reader/writer roles. No account keys or subscription IDs belong in git.
Configure a protected `azure-approved` GitHub Environment with required reviewers;
the repository now makes training/CD manual. Do not dispatch legacy workflows
without reviewing their older deployment contracts.

Continue with [storage](azure_storage.md), [training](azure_ml_training.md),
[deployment](azure_deployment.md), and [cost control](azure_cost_optimization.md).
