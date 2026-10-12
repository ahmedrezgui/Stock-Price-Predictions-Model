param(
    [Parameter(Mandatory=$true)][string]$ImageTag,
    [string]$ResourceGroup = 'rg-stock-forecast-student',
    [string]$RegistryName = 'stockforecastrezgu20261012',
    [string]$StorageAccount = 'ststockhistoricdata'
)
$ErrorActionPreference = 'Stop'
$env:AZURE_RESOURCE_GROUP = $ResourceGroup
$env:AZURE_STORAGE_ACCOUNT_URL = "https://$StorageAccount.blob.core.windows.net"
$env:FORECAST_REGISTRY_CONTAINER = 'forecast-registry'
if (-not $env:MLFLOW_TRACKING_URI) { throw 'Set MLFLOW_TRACKING_URI and Azure ML workspace environment variables first.' }
if ((Get-FileHash 'deployment/initial-release.json').Hash -ne (Get-FileHash 'configs/initial_release.json').Hash) {
    throw 'Packaged approval differs from the owner-approved manifest.'
}
& .venv/Scripts/python.exe -m stock_forecasting.initial_release reserve --mlflow-uri $env:MLFLOW_TRACKING_URI
if ($LASTEXITCODE -ne 0) { throw 'Initial release reservation failed. Nothing deployed.' }
$identity = az identity show -g $ResourceGroup -n stock-forecast-api-id -o json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Cannot read serving identity.' }
$private = Join-Path $env:LOCALAPPDATA 'StockForecast'
New-Item -ItemType Directory -Path $private -Force | Out-Null
$keyPath = Join-Path $private 'api-key.txt'
if (-not (Test-Path -LiteralPath $keyPath)) {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($bytes)
    $rng.Dispose()
    [Convert]::ToBase64String($bytes) | Set-Content -LiteralPath $keyPath -NoNewline
}
$parameters = @{
    location = @{value='uksouth'}
    appName = @{value='stock-forecast-api'}
    environmentName = @{value='stock-forecast-preview-env'}
    image = @{value="$RegistryName.azurecr.io/stock-forecast:$ImageTag"}
    deploymentId = @{value=('initial-' + (Get-Date -Format 'yyyyMMddHHmmss'))}
    identityId = @{value=$identity.id}
    identityClientId = @{value=$identity.clientId}
    registryServer = @{value="$RegistryName.azurecr.io"}
    storageAccountUrl = @{value=$env:AZURE_STORAGE_ACCOUNT_URL}
    monitorContainer = @{value='forecast-monitoring'}
    apiKey = @{value=(Get-Content -LiteralPath $keyPath -Raw).Trim()}
}
$parameterPath = Join-Path $private 'api-parameters.json'
try {
    @{parameters=$parameters} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $parameterPath
    $url = az deployment group create --name stock-forecast-initial --resource-group $ResourceGroup --template-file infrastructure/azure/container-app.bicep --parameters "@$parameterPath" --query properties.outputs.url.value -o tsv
    if ($LASTEXITCODE -ne 0) { throw 'Azure deployment failed. Deployed-model records have not been changed.' }
    $url | Set-Content -LiteralPath (Join-Path $private 'api-url.txt')
    Write-Output "API URL: $url"
    Write-Output "API key file: $keyPath"
    Write-Output 'Run the live checks and initial_release confirm before declaring the release complete.'
} finally {
    Remove-Item -LiteralPath $parameterPath -ErrorAction SilentlyContinue
}
