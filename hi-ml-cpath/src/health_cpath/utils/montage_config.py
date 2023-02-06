#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Optional

import param


class MontageConfig(param.Parameterized):
    dataset = \
        param.String(default="",
                     doc="The name of the AzureML dataset to use for creating the montage. The dataset will be "
                         "mounted automatically. Use an absolute path to a folder on the local machine to bypass "
                         "mounting.")
    datastore = \
        param.String(default="",
                     doc="The name of the AzureML datastore where the dataset is defined.")
    conda_env: Optional[Path] = \
        param.ClassSelector(class_=Path, default=Path("hi-ml/hi-ml-cpath/environment.yml"), allow_None=True,
                            doc="The Conda environment file that should be used when submitting the present run to "
                                "AzureML. If not specified, the hi-ml-cpath environment file will be used.")
    level: int = \
        param.Integer(default=1,
                      doc="Resolution downsample level, e.g. if lowest resolution is 40x and the available "
                          "downsample levels are [1.0, 4.0, 16.0] then level = 1 corresponds to 10x magnification")
    cluster: str = \
        param.String(default="", allow_None=False,
                     doc="The name of the GPU or CPU cluster inside the AzureML workspace"
                         "that should execute the job. To run on your local machine, omit this argument.")
    exclude_by_slide_id: Optional[Path] = \
        param.ClassSelector(class_=Path, default=None, allow_None=True,
                            doc="Provide a file that contains slide IDs that should be excluded. File format is "
                                "CSV, the first column is used as the slide ID. If the file is empty, no slides "
                                "will be excluded.")
    include_by_slide_id: Optional[Path] = \
        param.ClassSelector(class_=Path, default=None, allow_None=True,
                            doc="Provide a file that contains slide IDs that should be included. File format is "
                                "CSV, the first column is used as the slide ID. If the file is empty, no montage "
                                "will be produced.")
    image_glob_pattern: str = \
        param.String(default="",
                     doc="When provided, use this pattern in rglob to find the files that should be included in the "
                         "montage. Example: '**/*.tiff' to find all TIFF files recursive. You may have to escape "
                         "the pattern in your shell.")
    width: int = \
        param.Integer(default=60_000,
                      doc="The width of the montage in pixels")
    output_path: Path = \
        param.ClassSelector(class_=Path,
                            default=Path("outputs"),
                            doc="The folder where the montage will be saved")
    parallel: int = \
        param.Integer(default=8,
                      doc="The number of parallel processes to use when creating the montage.")
    backend: str = \
        param.String(default="openslide",
                     doc="The backend to use for reading the slides. Can be 'openslide' or 'cucim'")
