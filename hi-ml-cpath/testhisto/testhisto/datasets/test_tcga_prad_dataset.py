#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os

import pytest

from health_cpath.datasets.default_paths import TCGA_PRAD_DATASET_DIR
from health_cpath.datasets.tcga_prad_dataset import TcgaPradDataset
from health_cpath.utils.naming import SlideKey


@pytest.mark.skipif(not os.path.isdir(TCGA_PRAD_DATASET_DIR),
                    reason="TCGA-PRAD dataset is unavailable")
def test_dataset() -> None:
    dataset = TcgaPradDataset(TCGA_PRAD_DATASET_DIR)

    expected_length = 449
    assert len(dataset) == expected_length

    expected_num_positives = 10
    assert dataset.dataset_df[dataset.label_column].sum() == expected_num_positives

    sample = dataset[0]
    assert isinstance(sample, dict)

    expected_keys = [SlideKey.IMAGE, SlideKey.IMAGE_PATH, SlideKey.LABEL, SlideKey.METADATA]
    assert all(key in sample for key in expected_keys)

    image_path = sample[SlideKey.IMAGE_PATH]
    assert isinstance(image_path, str)
    assert os.path.isfile(image_path)
