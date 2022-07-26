# DeepMIL model for genetic mutation on TCGA-CRCK

## Background

The repository contains the configuration for training a Deep Multiple Instance Learning (DeepMIL) models for microsatellite instability (MSI) prediction on the [TCGA-CRCk dataset](https://zenodo.org/record/2530835).  Instructions to download and prepare the dataset are [here](public_datasets.md). The dataset is composed of tiles.

The models are developed to reproduce the results described in the DeepSMILE papers
[Schirris et al. 2021](https://pubmed.ncbi.nlm.nih.gov/35596966/) and [Schirris et al. 2022](https://www.sciencedirect.com/science/article/abs/pii/S1361841522001116).

A ResNet18 encoder that was pre-trained on ImageNet is downloaded on-the-fly (from
[here](https://download.pytorch.org/models/) at the start of the training run when using the ImageNet configuration. The SSL MIL configuration requires the checkpoint of a pre-trained SSL encoder (ResNet50) on the TCGA-CRCk dataset. The SSL encoder that can be obtained following the instructions on [how to train a custom SSL encoder on a pre-tiled dataset](ssl_on_tile_dataset.md).

## Preparations

Please follow the instructions in the [Readme file](../README.md#setting-up-python) to create a Conda environment and
activate it, and the instructions to [set up Azure](../README.md#setting-up-azureml) to run in the cloud.

## Running the model

You can run the model in the same way you run the [benchmark model on PANDA](panda_model.md). If you have GPU available locally, you can run training on that machine, by executing in `<root>/hi-ml-histopathology`:

```shell
conda activate HimlHisto
python ../hi-ml/src/health_ml/runner.py --model health_cpath.TcgaCrckImageNetMIL
```

If you setup a GPU cluster in Azure, you can run the training in the cloud by simply appending name of the compute cluster:

```shell
conda activate HimlHisto
python ../hi-ml/src/health_ml/runner.py --model health_cpath.TcgaCrckImageNetMIL --cluster=<your_cluster_name>
```

Assuming you pre-trained your own SSL encoder and updated the checkpoint path in the SSLencoder class, the SSL MIL model configuration can be run using the flag `--model histopathology.TcgaCrckSSLMIL`.

## Expected results
ImageNet MIL (ResNet18)
- Test AUROC: 0.706 ± 0.041
- Test F1 score: 0.490 ± 0.041

SSL MIL (ResNet50)
- Test AUROC: 0.825 ± 0.065
- Test F1 score: 0.516 ± 0.171
