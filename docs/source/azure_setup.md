# Setting up Azure

## Creating an AzureML workspace

- [Create a resource
  group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal)
- [Create a storage account for
  datasets](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)
- [Create an AzureML workspace, compute instances and compute clusters](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources)

## Accessing the workspace

- Clone the `hi-ml` repository to your VM / compute instance
- In the browser, navigate to the AzureML workspace that you want to use for running your training job.
- In the top right section, there will be a dropdown menu showing the name of your AzureML workspace. Expand that.
- In the panel, there is a link "Download config file". Click that.
- This will download a file `config.json`. Copy that file to the root folder of your repository on your VM / compute
  instance.
