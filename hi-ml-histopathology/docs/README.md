# Using the Histopathology Models

## Onboarding to Azure

Please follow the [detailed instructions here](azure_setup.md).

## Creating datasets

We are working with two public datasets, [PANDA](https://panda.grand-challenge.org/) and
[TCGA-Crck](https://zenodo.org/record/2530835).

Please follow the [detailed instructions](public_datasets.md) to download and prepare the datasets in Azure.

## Training a model

* [Train a DeepMil model with an ImageNet encoder on the PANDA dataset](panda_model.md)
* [Train a TCGA-Crck DeepMil on tile dataset with ImageNet encoder](tcga-crck_model.md)
  * How to invoke training on TCGA-Crck data, using the existing model config
  * Where to find the results and how to interpret them: AzureML dashboards, HTML report, checkpoints
  * What results are expected (accuracy)
  * If people want to re-use that model on their data:
    * What is the expected data format?
    * Which parameters need to be tuned?

## Training an SSL encoder

* [Train an SSL encoder on a pre-tiled dataset: TCGA-Crck](ssl_on_tile_dataset.md)
* How to use the checkpoint in a classifier
* If people want to train an SSL model on their data:
  * What is the expected data format?
  * Which parameters need to be tuned?

## Visualizing data and results in DSA

* [Setting up DSA](dsa_setup.md)
  * How to deploy DSA in Azure
  * How to link blob storage with DSA
  * How to get an API token for DSA and put it into AzureML's KeyVault
* [Using DSA to look at data and model results](dsa_usage.md)
  * Visualize the previously uploaded data in DSA
  * Show how model heatmaps go into DSA
