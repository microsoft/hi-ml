# Setting up Azure

If you already have an AzureML workspace available, you can go straight to [the last step](#accessing-the-workspace).

To set up Azure, you need to have:

- A subscription to Azure.
- An account that has "Owner" permissions to the Azure subscription.

There are two ways to set up all necessary resources, either via the Azure portal or via the Azure Commandline Interface (CLI).
We recommend the CLI because all necessary resources can be easily created via a single script.

## Creating an AzureML workspace via the Azure Portal

If you prefer to create your workspace via the web UI on the Azure Portal, please follow the steps below.

- [Create a resource
  group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal)
- [Create a storage account for
  datasets](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)
- [Create an AzureML workspace, compute instances and compute clusters](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources)

## Creating an AzureML workspace via the Azure Commandline Tools

A pureley commandline-drive setup is possible via the [Azure Commandline Tools](https://docs.microsoft.com/en-us/cli/azure/). These tools are available for multiple platforms, including Linux, Mac, and Windows.

After downloading the commandline tools, you can run the following command add the `ml` extension that is required to create an AzureML workspace:

```bash
az extension add --name ml
```

### Documentation

- [Creating AzureML workspaces via the CLI](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace-cli)
- [Creating Azure storage accounts via the CLI](https://docs.microsoft.com/en-us/cli/azure/storage?view=azure-cli-latest)
- [Schema for creating datastores](https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-datastore-blob)
- [Creating datastores via the CLI](https://docs.microsoft.com/en-us/cli/azure/ml/datastore?view=azure-cli-latest)
- [Create a shared access signature](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blob-user-delegation-sas-create-cli)

### Collecting the necessary information

Find out which Azure datacenter locations you can use:

```bash
az account list-locations -o table
```

You will need the location names (second column in the table) to create resources in the right geographical regions. Choosing the right region can be particularly important if your data governance requires the data to be processed inside certain geographical boundaries.

For the storage account, please choose an SKU from based on your needs as described [here](https://docs.microsoft.com/en-us/rest/api/storagerp/srp_sku_types). Most likely, `Standard_LRS` will be the right SKU for you.

#### Creating resources

In the script below, you will need to replace the values of the following variables:

- `location` - the location of the Azure datacenter you want to use
- `prefix` - the prefix you want to use for the resources you create. This will also be the name of the AzureML workspace.

```bash
export location=uksouth     # The Azure location where the resources should be created
export prefix=himl          # The name of the AzureML workspace, and prefix for all other resources
export container=datasets
export datastorefile=datastore.yaml
az group create \
    --name ${prefix}rg \
    --location ${location}
az storage account create \
    --name ${prefix}data \
    --resource-group ${prefix}rg \
    --location ${location} \
    --sku Standard_LRS
az storage container create \
    --account-name ${prefix}data \
    --name ${container} \
    --auth-mode
az ml workspace create \
    --resource-group ${prefix}rg \
    --name ${prefix} \
    --location ${location}
key=$(az storage account keys list --resource-group ${prefix}rg --account-name ${prefix}data --query [0].value -o tsv)cat >${datastorefile} <<EOL
\$schema: https://azuremlschemas.azureedge.net/latest/azureBlob.schema.json
name: datasets
type: azure_blob
description: Pointing to the datasets container in the ${prefix}data storage account.
account_name: ${prefix}data
container_name: ${container}
credentials:
  account_key: ${key}
EOL
az ml datastore create --file ${datastorefile} --resource-group ${prefix}rg --workspace-name ${prefix}
rm ${datastorefile}
```

Note that the datastore will use the storage account key to authenticate. You may want to switch that to Shared Access Signature (SAS).

Now that you have created the core AzureML workspace, you need to
[create a compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#cluster).

The final step is to download the workspace configuration file, as descreibed [here](#accessing-the-workspace).

### Adjusting permissions

- Find the AzureML workspace that you just created in the Azure Portal.
- Add yourself and your team members with "Contributor" permissions to the workspace, following the guidelines [here](https://docs.microsoft.com/en-us/azure/role-based-access-control/role-assignments-portal?tabs=current).

## Accessing the workspace

- Clone the `hi-ml` repository to your VM / compute instance
- In the browser, navigate to the AzureML workspace that you want to use for running your training job.
- In the top right section, there will be a dropdown menu showing the name of your AzureML workspace. Expand that.
- In the panel, there is a link "Download config file". Click that.
- This will download a file `config.json`. Copy that file to the root folder of your repository on your VM / compute
  instance.
