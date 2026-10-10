# Student-budget cost controls

No resource prices or subscription eligibility were measured for your account.
Use current regional/currency prices, remaining student credit and actual quota
before approval. Microsoft pricing pages were checked on 2026-10-09:
[Azure ML](https://azure.microsoft.com/en-us/pricing/details/machine-learning/),
[Container Apps](https://azure.microsoft.com/en-us/pricing/details/container-apps/),
[Blob Storage](https://azure.microsoft.com/en-us/pricing/details/storage/blobs/).

| Component | Need | Cost model and idle behavior |
|---|---|---|
| Local Python/CPU/GPU | Required for this verified workflow | No Azure consumption |
| Blob LRS | Cloud dataset/artifact persistence | GB-month + versions + transactions + egress; storage persists while idle |
| Azure ML workspace | Optional jobs/registry/tracking | Dependent storage, Key Vault, monitoring and registry may bill |
| CPU training cluster min=0, max=1 | Optional remote training | VM runtime + supporting resources; scale down after 120 seconds |
| Container Apps Consumption min=0 | Optional demo API | Active/idle replica seconds + requests; cold starts; supporting services still bill |
| Managed online endpoint | Optional, budget-sensitive | Provisioned deployment compute remains allocated without traffic |
| Existing App Service | Reuse only after inspection | The plan can continue billing after an app is stopped |
| GPU cluster | Optional | Verify eligible family quota; never assume credits grant access |

Estimate a training run as VM hourly rate * measured runtime hours plus data,
build and logging. Estimate an always-on endpoint as roughly 730 * hourly rate *
instance count per month, then add dependencies and transfer. Estimate a
consumption service using allocated vCPU/GiB * measured active/idle seconds and
requests minus applicable shared monthly grants. Do not double-count a grant
already consumed by another app. Use the [pricing calculator](https://azure.microsoft.com/en-us/pricing/calculator/)
with your chosen region; there is deliberately no invented universal dollar quote.

Before provisioning, set resource tags, one-node limits, job timeout and a spending
budget well below remaining credit. In Cost Management choose the subscription
or resource-group scope, create a monthly budget with 50/80/100% alerts where the
offer supports budgets, and check actual/forecast spend. Alerts are not hard caps;
the student spending limit is a separate subscription feature. Review the
[cost planning guide](https://learn.microsoft.com/en-us/azure/machine-learning/concept-plan-manage-cost?view=azureml-api-2)
for supported account types and dependent charges.

Read-only checks:

```powershell
az ml compute list -o table
az ml online-endpoint list -o table
az resource list -g $env:AZURE_RESOURCE_GROUP -o table
az ml job list --max-results 20 -o table
```

After separate explicit approval, stop a named compute instance with
`az ml compute stop --name <instance>`. For a cluster, retain min-instances=0 and
cancel unwanted jobs with `az ml job cancel --name <job>`; confirm node count drops.
Managed endpoint cleanup is deletion, not an assumed stop/start API:

```powershell
# Destructive examples only; verify subscription, resource group and exact names first.
az ml online-deployment delete --name '<idle-deployment>' --endpoint-name '<endpoint>'
az ml online-endpoint delete --name '<unused-endpoint>'
```

Do not delete shared resource groups or storage containing lineage needed by
retained models. Inventory each candidate with `az resource show --ids <id>`,
export required manifests/artifacts and obtain approval before `az resource delete
--ids <id>`. Container Apps min=0 handles idle replicas; remove unused revisions,
registries and log workspaces only after checking dependencies. Blob lifecycle
rules need a retention policy that preserves referenced versions. Review costs
again after cleanup; soft deletion and retained versions can still occupy storage.
[Azure ML cost optimization](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-optimize-cost?view=azureml-api-2)
