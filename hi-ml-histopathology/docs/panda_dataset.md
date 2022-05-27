# Datasets

## Setting up your storage account and tools

Your Azure account needs to have permissions to write to the storage account. You need to have, for example,
"Storage Blob Data Contributor" permissions.

Your storage account should have a container called `datasets`. Verify in the Azure Portal that this container exists:
You can do that by going to the "Containers" section in the left-hand navigation. If there is no such container, create
one with the "+ Container" button.

To upload the PANDA dataset to Azure, we recommend using [azcopy](http://aka.ms/azcopy). You will first need to log in,
by calling `azcopy login`. Follow the instructions at the prompt.

## PANDA dataset

The PANDA dataset was released with the [Prostate cANcer graDe Assessment (PANDA)
Challenge](https://panda.grand-challenge.org/). The dataset is available from
[Kaggle](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data). To get access to the dataset, you need to
register with Kaggle, then press "Download all". This will download a 200GB ZIP file.

### Uploading

Now unzip the ZIP file. Then rename the folder in which the files reside to `PANDA`. To double-check, there should now
be a file `PANDA/train.csv` in your current directory.

```shell
head PANDA/train.csv  # just to check if we are in the right folder
azcopy copy PANDA https://<your_storage_account>.blob.core.windows.net/datasets/ --recursive
```

## TCGA-Crck dataset

```shell
wget https://zenodo.org/record/2530835/files/CRC_DX_TEST_MSIMUT.zip?download=1
wget https://zenodo.org/record/2530835/files/CRC_DX_TEST_MSIMUT.zip?download=1
wget https://zenodo.org/record/2530835/files/CRC_DX_TEST_MSIMUT.zip?download=1
wget https://zenodo.org/record/2530835/files/CRC_DX_TEST_MSIMUT.zip?download=1
```

## Making your storage account accessible to AzureML

As a last step, you need to ensure that AzureML has access to your storage account. For that, you need to create a
datastore [following the instructions
here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-datastore?tabs=cli-identity-based-access%2Ccli-adls-identity-based-access%2Ccli-azfiles-account-key%2Ccli-adlsgen1-identity-based-access).
