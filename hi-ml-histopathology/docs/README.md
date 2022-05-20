# Using the histopathology models

## Onboarding to Azure

To be completed later, this is not priority 0.

## Creating datasets

* [How to get PANDA (whole slides) and upload it to Azure](panda_dataset.md)
* [How to get TCGA-Crck (tiles) and upload it to Azure](tcga-crck_dataset.md)

## Training a model

* [PANDA DeepMil on whole slides with ImageNet encoder](panda_model.md)
  * How to invoke training for the existing model on PANDA data
  * Where to find results and how to interpret them
  * What results are expected (accuracy)
  * If people want to re-use that model on their data:
    * What is the expected data format?
    * Which parameters need to be tuned?
* [TCGA-Crck DeepMil on tile dataset with ImageNet encoder](tcga-crck_model.md)
  * How to invoke training for the existing model on TCGA-Crck data
  * Where to find results and how to interpret them
  * What results are expected (accuracy)
  * If people want to re-use that model on their data:
    * What is the expected data format?
    * Which parameters need to be tuned?

## Training an SSL encoder

* [Train an SSL encoder on a pre-tiled dataset: TCGA-Crck](ssl_tile_dataset.md)
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
