# HI-ML Tools for Computational Pathology

The directory [`hi-ml-histopathology`](https://github.com/microsoft/hi-ml/tree/main/hi-ml-histopathology) contains code
for runnning experiments in Computational Pathology.

## Setting up your computer

The tools for computational pathology cannot be as a Python package, but rather directly from the Git repository. Please
follow the instructions in [README](https://github.com/microsoft/hi-ml/blob/main/hi-ml-histopathology/README.md) to set
up your local Python environment.

## Onboarding to Azure

Please follow the [instructions here](azure_setup.md).

## Creating datasets

In our example models, we are working with two public datasets, [PANDA](https://panda.grand-challenge.org/) and
[TCGA-Crck](https://zenodo.org/record/2530835).

Please follow the [detailed instructions](public_datasets.md) to download and prepare these datasets in Azure.

## Training models

- [Train a DeepMIL model with an ImageNet encoder on the PANDA dataset (whole slides)](panda_model.md)
- [Train an SSL encoder on the TCGA-Crck dataset (tiles)](ssl_on_tile_dataset.md)

## Visualizing data and results in Digital Slide Archive DSA

- [Setting up DSA](./dsa.md#azure-deployment)
- [Using DSA to look at data and model results](./dsa.md#visualizing-azure-machine-learning-results)

## New Model configurations

To define your own model configuration, place a class definition in the directory `histopathology.configs`. The class should
inherit from a
[LightningContainer](https://github.com/microsoft/hi-ml/blob/39911d217c919d8213ad36c9c776f69369d98509/hi-ml/src/health_ml/lightning_container.py#L24).
As an example, please check the [HelloWorld
model](https://github.com/microsoft/hi-ml/blob/0793cbd1a874920d04b0a8f1298a7a112cfd712c/hi-ml/src/health_ml/configs/hello_world.py#L232)
or the [base class for the MIL
models](https://github.com/microsoft/hi-ml/blob/1d96c9bcdb326ad4d145ab082f45a2116d776a76/hi-ml-histopathology/src/histopathology/configs/classification/BaseMIL.py#L39).

## Mount datasets

If you would like to inspect or analyze the datasets that are stored in Azure Blob Storage, you can either download them
or mount them. "Mounting" here means that the dataset will be loaded on-demand over the network (see also [the
docs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-with-datasets#mount-vs-download)). This is ideal if you expect that
you will only need a small number of files, or if the disk of your machine is too small to download the full dataset.

You can mount the dataset by executing this script in `<root>/hi-ml-histopathology`:

```shell
python src/histopathology/scripts/mount_azure_dataset.py --dataset_id PANDA
```

After a few seconds, this may bring up a browser to authenticate you in Azure, and let you access the AzureML
workspace that you chose by downloading the `config.json` file. If you get an error message saying that authentication
failed (error message contains "The token is not yet valid (nbf)"), please ensure that your
system's time is set correctly and then try again. On WSL, you can use `sudo hwclock -s`.

Upon success, the script will print out:

```text
Dataset PANDA will be mounted at /tmp/datasets/PANDA.
```
