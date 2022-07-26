#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
from health_cpath.datasets.default_paths import TCGA_CRCK_DATASET_ID

from health_cpath.utils.tcga_utils import extract_fields
from health_cpath.datasets.tcga_prad_dataset import TcgaPradDataset


def check_dataset_csv_paths(dataset_dir: Path) -> None:
    df = pd.read_csv(dataset_dir / TcgaPradDataset.DEFAULT_CSV_FILENAME)
    for img_path in df.image:
        assert (dataset_dir / img_path).is_file()


if __name__ == '__main__':
    # Script needs to be started in the parent folder of the dataset folder
    current_dir = Path.cwd()
    expected_datasetdir = TCGA_CRCK_DATASET_ID
    if not (current_dir / expected_datasetdir).is_dir:
        raise ValueError(f"The current folder must contain the actual dataset folder {expected_datasetdir}")
    dataset_dir = current_dir / expected_datasetdir
    expected_subdirs = ["CRC_DX_TEST", "CRC_DX_TRAIN"]
    if not all([(dataset_dir / subdir).is_dir() for subdir in expected_subdirs]):
        raise ValueError(f"The folder {expected_datasetdir} needs to have these subfolder: {expected_subdirs}")
    image_paths = [str(image_path.relative_to(dataset_dir))
                   for split_dir in dataset_dir.iterdir()
                   for class_dir in split_dir.iterdir()
                   for image_path in class_dir.iterdir()]

    df = pd.DataFrame(image_paths, columns=['image'])

    # takes up to ~20 seconds
    df = df.apply(extract_fields, axis='columns', result_type='expand')
    df.to_csv(dataset_dir / TcgaPradDataset.DEFAULT_CSV_FILENAME, index=False)

    check_dataset_csv_paths(dataset_dir)
