param(
    [string]$ResourceGroup = 'rg-stock-forecast-student',
    [string]$Location = 'uksouth',
    [string]$AppName = 'stock-forecast-preview',
    [string]$EnvironmentName = 'stock-forecast-preview-env',
    [string]$RegistryName = 'stockforecastrezgu20261012',
    [string]$ImageTag = '20261012-1',
    [string]$DeploymentId = ('preview-' + (Get-Date -Format 'yyyyMMddHHmmss')),
    [string]$IdentityName = 'stock-forecast-preview-id',
    [string]$StorageAccount = 'ststockhistoricdata'
)
$ErrorActionPreference = 'Stop'
if ($AppName -notlike '*preview*') { throw 'This script only deploys a preview app.' }
$preview = Get-Content -LiteralPath 'deployment/preview.json' -Raw | ConvertFrom-Json
if ($preview.production_approved -ne $false) { throw 'Expected an explicitly unapproved preview package.' }
$identity = az identity show -g $ResourceGroup -n $IdentityName -o json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Cannot read preview identity.' }
$privateDirectory = Join-Path $env:LOCALAPPDATA 'StockForecast'
New-Item -ItemType Directory -Path $privateDirectory -Force | Out-Null
$keyPath = Join-Path $privateDirectory 'preview-api-key.txt'
if (-not (Test-Path -LiteralPath $keyPath)) {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($bytes)
    $rng.Dispose()
    [Convert]::ToBase64String($bytes) | Set-Content -LiteralPath $keyPath -NoNewline
}
$parameters = @{
    location = @{value=$Location}
    appName = @{value=$AppName}
    environmentName = @{value=$EnvironmentName}
    image = @{value="$RegistryName.azurecr.io/stock-forecast-preview:$ImageTag"}
    deploymentId = @{value=$DeploymentId}
    identityId = @{value=$identity.id}
    identityClientId = @{value=$identity.clientId}
    registryServer = @{value="$RegistryName.azurecr.io"}
    storageAccountUrl = @{value="https://$StorageAccount.blob.core.windows.net"}
    monitorContainer = @{value='forecast-preview-monitoring'}
    apiKey = @{value=(Get-Content -LiteralPath $keyPath -Raw).Trim()}
}
$parameterPath = Join-Path $privateDirectory 'preview-parameters.json'
try {
    @{parameters=$parameters} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $parameterPath
    $url = az deployment group create --name stock-forecast-preview --resource-group $ResourceGroup --template-file infrastructure/azure/container-app.bicep --parameters "@$parameterPath" --query properties.outputs.url.value -o tsv
    if ($LASTEXITCODE -ne 0) { throw 'Azure deployment failed; production state was not changed.' }
    $url | Set-Content -LiteralPath (Join-Path $privateDirectory 'preview-url.txt')
    Write-Output "Preview URL: $url"
    Write-Output "API key file: $keyPath"
} finally {
    Remove-Item -LiteralPath $parameterPath -ErrorAction SilentlyContinue
}
