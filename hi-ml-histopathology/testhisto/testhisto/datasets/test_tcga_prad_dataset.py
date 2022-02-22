#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os

import pytest
import torch
from monai.data.dataset import Dataset
from torch.utils.data import DataLoader

from histopathology.datasets.default_paths import TCGA_PRAD_DATASET_DIR, TCGA_PRAD_TILES_DATASET_DIR
from histopathology.datasets.tcga_prad_dataset import TcgaPradDataset
from histopathology.datasets.tcga_prad_tiles_dataset import TcgaPrad_TilesDataset
from histopathology.models.transforms import LoadTiled


@pytest.mark.skipif(not os.path.isdir(TCGA_PRAD_DATASET_DIR), reason="TCGA-PRAD dataset is unavailable")
def test_dataset() -> None:
    dataset = TcgaPradDataset(TCGA_PRAD_DATASET_DIR)

    expected_length = 449
    assert len(dataset) == expected_length

    expected_num_positives = 10
    assert dataset.dataset_df[dataset.LABEL_COLUMN].sum() == expected_num_positives

    sample = dataset[0]
    assert isinstance(sample, dict)

    expected_keys = [dataset.SLIDE_ID_COLUMN, dataset.CASE_ID_COLUMN, dataset.IMAGE_COLUMN, dataset.LABEL_COLUMN]
    assert all(key in sample for key in expected_keys)

    image_path = sample[dataset.IMAGE_COLUMN]
    assert isinstance(image_path, str)
    assert os.path.isfile(image_path)


@pytest.mark.skipif(not os.path.isdir(TCGA_PRAD_TILES_DATASET_DIR), reason="TCGA-PRAD tiles dataset is unavailable")
def test_tiles_dataset() -> None:
    dataset = TcgaPrad_TilesDataset(TCGA_PRAD_TILES_DATASET_DIR)

    expected_length = 1993410
    assert len(dataset) == expected_length

    sample_dataset = Dataset(dataset[0:2], transform=LoadTiled("image"))  # type: ignore
    sample = sample_dataset[0]
    expected_keys = [
        dataset.SLIDE_ID_COLUMN,
        dataset.IMAGE_COLUMN,
        dataset.LABEL_COLUMN,
        dataset.TILE_X_COLUMN,
        dataset.TILE_Y_COLUMN,
    ]
    assert all(key in sample for key in expected_keys)
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 224, 224)

    batch_size = 16
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # type: ignore
    batch = next(iter(loader))
    assert all(key in batch for key in expected_keys)
    assert isinstance(batch["image"], torch.Tensor)
    assert batch["image"].shape == (batch_size, 3, 224, 224)
    assert batch["image"].dtype == torch.float32
    assert batch["label"].shape == (batch_size,)
    assert batch["label"].dtype == torch.int64
