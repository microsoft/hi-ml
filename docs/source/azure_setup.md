# Setting up Azure

If you already have an AzureML workspace available, you can go straight to [the last step](#accessing-the-workspace).

To set up all your Azure resources, you need to have:

- A subscription to Azure.
- An account that has "Owner" permissions to the Azure subscription.

There are two ways to set up all necessary resources, either via the Azure portal or via the Azure Command-line Interface (CLI).
We recommend the CLI because all necessary resources can be easily created via a single script.

## Creating an AzureML workspace via the Azure Portal

If you prefer to create your workspace via the web UI on the Azure Portal, please follow the steps below.

- [Create a resource
  group.](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal)
- [Create a storage account for
  datasets.](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)
- [Create an AzureML workspace, compute instances and compute clusters.](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources)

## Creating an AzureML workspace via the Azure Command-line Tools

A pureley command-line driven setup is possible via the [Azure Command-line Tools](https://docs.microsoft.com/en-us/cli/azure/). These tools are available for multiple platforms, including Linux, Mac, and Windows.

After downloading the command-line tools, you can run the following command add the `ml` extension that is required to create an AzureML workspace:

```bash
az extension add --name ml
```

### Documentation

- [Creating AzureML workspaces via the CLI.](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace-cli)
- [Creating Azure storage accounts via the CLI.](https://docs.microsoft.com/en-us/cli/azure/storage?view=azure-cli-latest)
- [Schema for creating datastores.](https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-datastore-blob)
- [Creating datastores via the CLI.](https://docs.microsoft.com/en-us/cli/azure/ml/datastore?view=azure-cli-latest)
- [Create a shared access signature.](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blob-user-delegation-sas-create-cli)

### Collecting the necessary information

Find out which Azure data centre locations you can use:

```bash
az account list-locations -o table
```

You will need the location names (second column in the table) to create resources in the right geographical regions. Choosing the right region can be particularly important if your data governance requires the data to be processed inside certain geographical boundaries.

For the storage account, please choose an SKU from based on your needs as described [here](https://docs.microsoft.com/en-us/rest/api/storagerp/srp_sku_types). Most likely, `Standard_LRS` will be the right SKU for you.

### Creating resources

The script below will create

- an AzureML workspace.
- a storage account to hold the datasets, with a container called `datasets`.
- a datastore that links the AzureML workspace to the storage account.
- a resource group for all of the above items.

In the script, you will need to replace the values of the following variables:

- `location` - the location of the Azure datacenter you want to use.
- `prefix` - the prefix you want to use for the resources you create. This will also be the name of the AzureML workspace.

```bash
export location=uksouth     # The Azure location where the resources should be created
export prefix=himl          # The name of the AzureML workspace. This is also the prefix for all other resources.
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
key=$(az storage account keys list --resource-group ${prefix}rg --account-name ${prefix}data --query [0].value -o tsv)
cat >${datastorefile} <<EOL
\$schema: https://azuremlschemas.azureedge.net/latest/azureBlob.schema.json
name: datasets
type: azure_blob
description: Pointing to the `${container}` container in the ${prefix}data storage account.
account_name: ${prefix}data
container_name: ${container}
credentials:
  account_key: ${key}
EOL
az ml datastore create --file ${datastorefile} --resource-group ${prefix}rg --workspace-name ${prefix}
rm ${datastorefile}
```

Note that the datastore will use the storage account key to authenticate. If you want to use Shared Access Signature (SAS) instead,
replace the creation of the datastore config file in the above script with the following command:

```bash
key=$(az storage container generate-sas --account-name ${prefix}data --name ${container} --permissions acdlrw --https-only --expiry 2024-01-01 -o tsv)
cat >${datastorefile} <<EOL
\$schema: https://azuremlschemas.azureedge.net/latest/azureBlob.schema.json
name: ${name}
type: azure_blob
description: Pointing to the `${container}` container in the ${prefix}data storage account.
account_name: ${prefix}data
container_name: ${container}
credentials:
  sas_token: ${key}
EOL
```

You can adjust the expiry date of the SAS token and the permissions of the SAS token (full read/write permission

in the script above). For further options, run `az storage container generate-sas --help`

### Creating compute clusters and permissions

Now that you have created the core AzureML workspace, you need to
[create a compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#cluster).

To adjust permissions, find the AzureML workspace that you just created in the Azure Portal. Add yourself and your team
members with "Contributor" permissions to the workspace, following the guidelines
[here](https://docs.microsoft.com/en-us/azure/role-based-access-control/role-assignments-portal?tabs=current).

## Accessing the workspace

The `hi-ml` toolbox relies on a workspace configuration file called `config.json` to access the right AzureML workspace.
This file can be downloaded from the UI of the workspace. It needs to be placed either in your copy of the `hi-ml` repository,
or in your repository that uses the `hi-ml` package.

- In the browser, navigate to the AzureML workspace that you want to use for running your training job.
- In the top right section, there will be a dropdown menu showing the name of your AzureML workspace. Expand that.
- In the panel, there is a link "Download config file". Click that.
- This will download a file `config.json`. Copy that file to the root folder of your repository.

The file `config.json` should look like this:

```json
{
  "subscription_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "resource_group": "myresourcegroup",
  "workspace_name": "myworkspace"
}
```

As an alternative to keeping the `config.json` file in your repository, you can specify the
necessary information in environment variables. The environment variables are:

- `HIML_SUBSCRIPTION_ID`: The subscription ID of the AzureML workspace, taken from the `subscription_id` field in the
  `config.json` file.
- `HIML_RESOURCE_GROUP`: The resource group of the AzureML workspace, taken from the `resource_group` field in the
  `config.json` file.
- `HIML_WORKSPACE_NAME`: The name of the AzureML workspace, taken from the `workspace_name` field in the `config.json`
  file.

When accessing the workspace, the `hi-ml` toolbox will first look for the `config.json` file. If it is not found, it
will fall back to the environment variables. For details, see the documentation of the `get_workspace` function in
[readthedocs](https://hi-ml.readthedocs.io/en/latest/api/health_azure.get_workspace.html).
