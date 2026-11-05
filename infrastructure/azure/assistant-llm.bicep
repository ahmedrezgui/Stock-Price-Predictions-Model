@description('New Azure OpenAI account name; never point this at an unrelated existing account.')
param accountName string
param location string = 'uksouth'
param deploymentName string = 'stock-assistant'
param modelName string = 'gpt-4.1-mini'
param modelVersion string = '2025-04-14'
@description('Deployment quota units, not a monetary spending cap.')
@minValue(1)
param capacity int = 10

resource account 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: accountName
  location: location
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: accountName
    publicNetworkAccess: 'Enabled'
  }
}

resource deployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: account
  name: deploymentName
  sku: { name: 'GlobalStandard', capacity: capacity }
  properties: {
    model: {
      format: 'OpenAI'
      name: modelName
      version: modelVersion
    }
    versionUpgradeOption: 'NoAutoUpgrade'
  }
}

output endpoint string = account.properties.endpoint
output assistantModel string = deployment.name
