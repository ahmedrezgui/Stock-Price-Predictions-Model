# Deployment and rollback — approval required

The standalone FastAPI service keeps inference independent of LangGraph and
unrelated credentials. Docker has three explicit targets: `base` for local
mounted artifacts/data, `packaged` for a reviewed deployment snapshot, and
`azureml` for the managed scoring server. Docker CLI was present, but the Linux
engine was unavailable during this session; **image build/run is unverified**.
No image was published and no live endpoint changed.

```powershell
docker build --target base -t stock-forecast:local .
docker run --rm -p 8000:8000 --mount "type=bind,source=$($PWD.Path)/artifacts/models,target=/app/artifacts/models,readonly" --mount "type=bind,source=$($PWD.Path)/data/raw,target=/app/data/raw,readonly" stock-forecast:local
Invoke-RestMethod http://127.0.0.1:8000/ready
```

The container runs as a nonroot user. For deployment, prepare `deployment/models`
with only the chosen exact version(s), and `deployment/data` with approved current
raw symbol CSVs. Do not package the whole experiment directory. Immutable image
tags/digests provide a reproducible artifact snapshot; freshness checks deliberately
fail after 14 days without refreshed data. This simple demo path uses a newly
reviewed image for each refresh; automated production data refresh is future work.

## Lower-idle-cost path: Container Apps Consumption

After reviewing data redistribution rights, build `--target packaged`, publish
to an approved registry and obtain an immutable image digest. Registry publication
is an external action requiring approval. The included Bicep accepts image, names,
region and a secure API key. It uses zero minimum / one maximum replica and HTTPS.
The template assumes an anonymously pullable image; private ACR requires registry
identity configuration and AcrPull before use. Never put private datasets in a
public image merely to satisfy that assumption.

```powershell
$image = '<approved-registry/image@sha256:digest>'
$appName = '<new-approved-app-name>'
$environmentName = '<approved-environment-name>'
# Supply apiKey through a secure deployment parameter mechanism, not a committed file.
az deployment group what-if -g $env:AZURE_RESOURCE_GROUP --template-file infrastructure/azure/container-app.bicep --parameters appName=$appName environmentName=$environmentName image=$image
az deployment group create -g $env:AZURE_RESOURCE_GROUP --template-file infrastructure/azure/container-app.bicep --parameters appName=$appName environmentName=$environmentName image=$image
az containerapp show -g $env:AZURE_RESOURCE_GROUP -n $appName --query properties.configuration.ingress.fqdn -o tsv
```

Secure parameter prompting/your secret injection mechanism supplies apiKey. Validate
health, readiness, unauthorized 401, authorized /models and /v1/predict before
sharing the URL. Configure log retention and inspect latency/status logs; cold
starts are expected with scale-to-zero. Container Apps bills resources and requests,
and scale-to-zero removes replica resource consumption charges, but associated
storage/registry/monitoring can still bill.
[Consumption billing](https://learn.microsoft.com/en-us/azure/container-apps/billing)

Rollback: record the previous approved image digest; after approval use
`az containerapp update -g <group> -n <app> --image <previous-digest>` and repeat
readiness/prediction checks. Keep the previous model's dataset and schema intact.

## Optional managed Azure ML endpoint

First check subscription quota, supported VM SKU, region and total monthly costs.
Build/publish `--target azureml`; its image includes the project and inference
server. Register the exact bundle, then explicitly create a candidate deployment:

```powershell
python -m stock_forecasting.azure deploy --name '<new-endpoint>' --model-id 'azureml:biat-return-lstm:<version>' --image '<approved-image-digest>' --instance-type '<approved-supported-SKU>' --approve-cloud-action
az ml online-deployment get-logs --name candidate --endpoint-name '<new-endpoint>' --lines 100
az ml online-endpoint invoke --name '<new-endpoint>' --deployment-name candidate --request-file request-observations.json
```

The endpoint uses `aad_token`. The scoring contract is completed canonical OHLCV
under `observations`; preprocessing and model loading match local artifacts.
Provisioning a candidate costs money even before traffic. The script does not
promote it. After explicit review of the invoked response, route traffic with
`az ml online-endpoint update --name <endpoint> --traffic candidate=100`.
Do not use this helper against an existing live endpoint without reviewing its
configuration first. Roll back traffic to a retained previous deployment, verify,
then separately approve deletion of the unsuccessful deployment. Keep deployment
tags for costs; inspect endpoint logs, failures and latency in Azure Monitor.
[Online endpoint authentication and requirements](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online?view=azureml-api-2)

The Bicep and SDK configurations are implemented but have not been validated
against an authenticated subscription. Resource approval, image builds, identity
configuration, regional quotas, network access, deployment and live smoke testing
remain explicit unexecuted steps.
