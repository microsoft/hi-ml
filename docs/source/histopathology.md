# Microsoft Health Intelligence Machine Learning Toolbox

## Overview

The directory `hi-ml-histopathology` contains code for runnning experiments in Computational Pathology.

## Datasets

There are classes for handling various histopathology datasets including:
- The [PANDA Challenge](https://www.computationalpathologygroup.eu/projects/panda-challenge/)
- [TGCA-PRAD](https://wiki.cancerimagingarchive.net/display/Public/TCGA-PRAD)
- [TGCA-Crck](https://zenodo.org/record/2530835#.YgE_IJbP1mM) (see <sup>1</sup>)

## Configs
This directory contains built-in configs for:

- DeepSMILE<sup>1</sup> and encoder specific variants, e.g.TcgaCrckImageNetMIL

To define your own config, it should be placed in the directory `histopathology.configs` and it should inherit from a [LightningContainer]() - see an example in `histopathology.configs.classification.BaseMIL.py`

1 Schirris (2021). DeepSMILE: Self-supervised heterogeneity-aware multiple instance learning for DNA
damage response defect classification directly from H&E whole-slide images. arXiv:2107.09405

## Creating tiles datasets

There is a script in `histopathology/preprocessing/tiling.py` for creating a tiles dataset from a slide dataset. This will load and process a slide, save tile images and save relevant information to a CSV file.

## Experiments with the himl-runner

Examples within the folder `histopathology.configs` can be run using the hi-ml runner.

For example, DeepSMILECrck is the container for experiments relating to DeepSMILE using the TCGA-CRCk dataset. Run using

```bash
himl-runner --model=hi-ml-histopathology.histopathology.configs.classification.TcgaCrckImageNetMIL [--azureml] [--cluster=<cluster name>]
```
