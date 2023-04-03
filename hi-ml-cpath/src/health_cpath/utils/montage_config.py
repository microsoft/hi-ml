#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import param

from health_azure.argparsing import create_argparser


class AzureRunConfig(param.Parameterized):
    cluster: str = param.String(
        default="",
        allow_None=False,
        doc="The name of the GPU or CPU cluster inside the AzureML workspace"
        "that should execute the job. To run on your local machine, omit this argument.",
    )
    datastore: str = param.String(default="", doc="The name of the AzureML datastore where the dataset is defined.")
    dataset: str = param.String(
        default="",
        doc="The name of the AzureML dataset to use for creating the montage. The dataset will be "
        "mounted automatically. Use an absolute path to a folder on the local machine to bypass "
        "mounting.",
    )
    conda_env: Optional[Path] = param.ClassSelector(
        class_=Path,
        default=Path("hi-ml/hi-ml-cpath/environment.yml"),
        allow_None=True,
        doc="The Conda environment file that should be used when submitting the present run to "
        "AzureML. If not specified, the hi-ml-cpath environment file will be used.",
    )
    wait_for_completion: bool = param.Boolean(
        default=False,
        doc="If True, wait for AML Run to complete before proceeding. If False, submit the run to AML and exit",
    )
    docker_shm_size: str = param.String("100g", doc="The shared memory in the Docker image for the AzureML VMs.")
    workspace_config_path: Optional[Path] = param.ClassSelector(
        class_=Path,
        default=None,
        allow_None=True,
        doc="The path to the AzureML workspace configuration file. If not specified, the "
        "configuration file in the current folder or one of its parents will be used.",
    )
    display_name: str = param.String(
        default="", doc="The display name of the AzureML run. If not specified, a default name will be used."
    )


class MontageConfig(AzureRunConfig):
    level: int = param.Integer(
        default=1,
        doc="Resolution downsample level, e.g. if lowest resolution is 40x and the available "
        "downsample levels are [1.0, 4.0, 16.0] then level = 1 corresponds to 10x magnification",
    )
    exclude_by_slide_id: Optional[Path] = param.ClassSelector(
        class_=Path,
        default=None,
        allow_None=True,
        doc="Provide a file that contains slide IDs that should be excluded. File format is "
        "CSV, the first column is used as the slide ID. If the file is empty, no slides "
        "will be excluded.",
    )
    include_by_slide_id: Optional[Path] = param.ClassSelector(
        class_=Path,
        default=None,
        allow_None=True,
        doc="Provide a file that contains slide IDs that should be included. File format is "
        "CSV, the first column is used as the slide ID. If the file is empty, no montage "
        "will be produced.",
    )
    image_glob_pattern: str = param.String(
        default="",
        doc="When provided, use this pattern in rglob to find the files that should be included in the "
        "montage. Example: '**/*.tiff' to find all TIFF files recursive. You may have to escape "
        "the pattern in your shell.",
    )
    width: int = param.Integer(default=60_000, doc="The width of the montage in pixels")
    output_path: Path = param.ClassSelector(
        class_=Path, default=Path("outputs"), doc="The folder where the montage will be saved"
    )
    parallel: int = param.Integer(default=8, doc="The number of parallel processes to use when creating the montage.")
    backend: str = param.String(
        default="openslide", doc="The backend to use for reading the slides. Can be 'openslide' or 'cucim'"
    )


def create_montage_argparser() -> ArgumentParser:
    return create_argparser(
        MontageConfig(),
        usage="python create_montage.py --dataset <dataset_folder> --image_glob_pattern '**/*.tiff' --width 1000",
        description="Create an overview image with thumbnails of all slides in a dataset.",
    )
