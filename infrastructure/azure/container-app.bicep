param location string = resourceGroup().location
param appName string
param environmentName string
param image string
@secure()
param apiKey string

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  properties: {}
}
resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      secrets: [{ name: 'forecast-api-key', value: apiKey }]
      ingress: { external: true, targetPort: 8000, transport: 'http', allowInsecure: false }
    }
    template: {
      containers: [{
        name: 'forecast'
        image: image
        env: [
          { name: 'FORECAST_ENV', value: 'production' }
          { name: 'FORECAST_API_KEY', secretRef: 'forecast-api-key' }
        ]
        resources: { cpu: json('1.0'), memory: '2Gi' }
        probes: [
          { type: 'Liveness', httpGet: { path: '/health', port: 8000 }, initialDelaySeconds: 30 }
          { type: 'Readiness', httpGet: { path: '/ready', port: 8000 }, initialDelaySeconds: 30 }
        ]
      }]
      scale: { minReplicas: 0, maxReplicas: 1 }
    }
  }
}
output url string = 'https://${app.properties.configuration.ingress.fqdn}'
