# How to use the repository examples

## Pre-requisites
- Azure subscription
  - https://azure.microsoft.com/en-us/
- Azure Cognitive Services Instance
  - https://azure.microsoft.com/en-us/products/cognitive-services/#overview
- Azure OpenAI Service Instance
  - https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/
  
## Deploy Azure OpenAI model
- https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal
<ol>
<li>Go to the Azure OpenAI Studio</li>
<li>Login with the resource you want to use</li>
<li>Select the Go to Deployments button under Manage deployments in your resource to navigate to the Deployments page</li>
<li>Create a new deployment called text-davinci-002 and choose the text-davinci-002 model from the drop-down.</li>
</ol>

## Fill in config parameters
- Open the config.cfg file
  - Replace the values in the file with the apikeys and model names of deployed services:
  - Example config:
```
[openai_api]
api_key:33XXXXXXXXXXXXXXXXXXXX2e
api_ep:https://XXXXX.openai.azure.com/
api_model:model_name
cog_svc_key:33XXXXXXXXXXXXXXXXXXXX2e
cog_svc_ep:https://XXXXX.cognitiveservices.azure.com

```
## Install requirements
- Install python packages in the [requirements.txt](requirements.txt) file.
## Navigate to example notebooks
- Open the sample notebooks using Jupyter to run in local or cloud environment.

  
