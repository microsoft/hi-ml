# Using the Histopathology Models

## Onboarding to Azure

Please follow the [detailed instructions here](azure_setup.md).

## Creating datasets

We are working with two public datasets, [PANDA](https://panda.grand-challenge.org/) and
[TCGA-Crck](https://zenodo.org/record/2530835).

Please follow the [detailed instructions](public_datasets.md) to download and prepare the datasets in Azure.

## Training a model

* [Train a DeepMIL model with an ImageNet encoder on the PANDA dataset (whole slides)](panda_model.md)
* [Train a DeepMIL model with an ImageNet encoder on the TCGA-Crck dataset (tiles)](tcga-crck_model.md)

## Training an SSL encoder

[Train an SSL encoder on the TCGA-Crck dataset (tiles)](ssl_on_tile_dataset.md)

## Visualizing data and results in DSA

* [Setting up DSA](dsa_setup.md)
  * How to deploy DSA in Azure
  * How to link blob storage with DSA
  * How to get an API token for DSA and put it into AzureML's KeyVault
* [Using DSA to look at data and model results](dsa_usage.md)
  * Visualize the previously uploaded data in DSA
  * Show how model heatmaps go into DSA
