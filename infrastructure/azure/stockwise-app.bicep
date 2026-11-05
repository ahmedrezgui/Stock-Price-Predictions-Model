param location string = 'uksouth'
param appName string = 'stockwise-app'
param environmentName string = 'stockwise-standard-env'
param registryServer string
param identityId string
param assistantImage string
param frontendImage string
param gatewayImage string
param storageAccountName string
@secure()
param passwordHash string
@secure()
param assistantSecrets object
param assistantEnv array
param microsoftAuthEnabled bool = false
param externalAccess bool = false
@minValue(0)
@maxValue(1)
param minReplicas int = 0

resource environment 'Microsoft.App/managedEnvironments@2026-07-01' = {
  name: environmentName
  location: location
  properties: {
    environmentMode: 'WorkloadProfiles'
    workloadProfiles: [{ name: 'Consumption', workloadProfileType: 'Consumption' }]
  }
}
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
    supportsHttpsTrafficOnly: true
  }
}
resource files 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' existing = { parent: storage, name: 'default' }
resource share 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: files
  name: 'stockwise-app'
  properties: { shareQuota: 5 }
}
resource mount 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: environment
  name: 'stockwise-files'
  properties: {
    azureFile: {
      accountName: storage.name
      accountKey: storage.listKeys().keys[0].value
      shareName: share.name
      accessMode: 'ReadWrite'
    }
  }
}
resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identityId}': {} } }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      secrets: concat(assistantSecrets.values, [{ name: 'app-password-hash', value: passwordHash }])
      registries: [{ server: registryServer, identity: identityId }]
      ingress: { external: externalAccess, targetPort: 8080, transport: 'http', allowInsecure: false }
    }
    template: {
      volumes: [{ name: 'persistent', storageType: 'AzureFile', storageName: mount.name, mountOptions: 'uid=1000,gid=1000,dir_mode=0770,file_mode=0660' }]
      containers: [
        {
          name: 'gateway'
          image: gatewayImage
          env: [{ name: 'STOCKWISE_HTPASSWD', secretRef: 'app-password-hash' }]
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
          probes: [{ type: 'Readiness', httpGet: { path: '/health', port: 8080 }, initialDelaySeconds: 5 }]
        }
        {
          name: 'frontend'
          image: frontendImage
          env: [
            { name: 'ASSISTANT_URL', value: 'http://localhost:8001' }
            { name: 'BACKEND_URL', value: 'http://localhost:8002' }
          ]
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
          probes: [{ type: 'Readiness', httpGet: { path: '/_stcore/health', port: 8501 }, initialDelaySeconds: 15 }]
        }
        {
          name: 'assistant'
          image: assistantImage
          env: concat(assistantEnv, [
            { name: 'ASSISTANT_MLFLOW_TRACKING_URI', value: 'http://localhost:5000/mlflow' }
            { name: 'MLFLOW_ENABLE_ASYNC_TRACE_LOGGING', value: 'false' }
            { name: 'ASSISTANT_DATA_ROOT', value: '/persistent/data' }
            { name: 'AGENT_MAX_RETRIES', value: '1' }
            { name: 'ASSISTANT_RECURSION_LIMIT', value: '25' }
          ])
          volumeMounts: [{ volumeName: 'persistent', mountPath: '/persistent' }]
          resources: { cpu: json('0.75'), memory: '1.5Gi' }
          probes: [{ type: 'Readiness', httpGet: { path: '/health', port: 8001 }, initialDelaySeconds: 30, failureThreshold: 10 }]
        }
        {
          name: 'dashboard'
          image: assistantImage
          command: ['uvicorn', 'apps.back.dashboard:app', '--host', '0.0.0.0', '--port', '8002']
          env: concat(assistantEnv, [{ name: 'ASSISTANT_DATA_ROOT', value: '/persistent/data' }])
          volumeMounts: [{ volumeName: 'persistent', mountPath: '/persistent' }]
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
        }
        {
          name: 'mlflow'
          image: assistantImage
          command: ['mlflow', 'server', '--backend-store-uri', 'file:///persistent/mlruns', '--artifacts-destination', '/persistent/artifacts', '--default-artifact-root', 'mlflow-artifacts:/', '--host', '0.0.0.0', '--port', '5000', '--workers', '1', '--static-prefix', '/mlflow', '--allowed-hosts', '*']
          env: [{ name: 'MLFLOW_ALLOW_FILE_STORE', value: 'true' }]
          volumeMounts: [{ volumeName: 'persistent', mountPath: '/persistent' }]
          resources: { cpu: json('0.5'), memory: '1Gi' }
        }
      ]
      scale: { minReplicas: minReplicas, maxReplicas: 1 }
    }
  }
}
resource auth 'Microsoft.App/containerApps/authConfigs@2024-03-01' = {
  parent: app
  name: 'current'
  properties: {
    platform: { enabled: microsoftAuthEnabled }
    httpSettings: { requireHttps: true }
  }
}
output url string = 'https://${app.properties.configuration.ingress.fqdn}'
