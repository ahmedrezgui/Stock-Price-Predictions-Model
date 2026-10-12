param location string = resourceGroup().location
param appName string
param environmentName string
param image string
param deploymentId string = uniqueString(image)
param identityId string
param identityClientId string
param registryServer string
param storageAccountUrl string
param dataContainer string = 'forecast-serving'
param monitorContainer string
@secure()
param apiKey string

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: environmentName
}
resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identityId}': {} }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      secrets: [{ name: 'forecast-api-key', value: apiKey }]
      registries: [{ server: registryServer, identity: identityId }]
      ingress: { external: true, targetPort: 8000, transport: 'http', allowInsecure: false }
    }
    template: {
      containers: [{
        name: 'forecast'
        image: image
        env: [
          { name: 'FORECAST_ENV', value: 'production' }
          { name: 'FORECAST_DEPLOYMENT_ID', value: deploymentId }
          { name: 'FORECAST_API_KEY', secretRef: 'forecast-api-key' }
          { name: 'AZURE_CLIENT_ID', value: identityClientId }
          { name: 'AZURE_STORAGE_ACCOUNT_URL', value: storageAccountUrl }
          { name: 'FORECAST_DATA_BACKEND', value: 'azure' }
          { name: 'FORECAST_DATA_CONTAINER', value: dataContainer }
          { name: 'FORECAST_MONITOR_BACKEND', value: 'azure' }
          { name: 'FORECAST_MONITOR_CONTAINER', value: monitorContainer }
        ]
        resources: { cpu: json('1.0'), memory: '2Gi' }
        probes: [
          { type: 'Liveness', httpGet: { path: '/health', port: 8000 }, initialDelaySeconds: 30 }
          { type: 'Readiness', httpGet: { path: '/ready', port: 8000 }, initialDelaySeconds: 60, timeoutSeconds: 30, periodSeconds: 60, failureThreshold: 3 }
        ]
      }]
      scale: { minReplicas: 0, maxReplicas: 1 }
    }
  }
}
output url string = 'https://${app.properties.configuration.ingress.fqdn}'
