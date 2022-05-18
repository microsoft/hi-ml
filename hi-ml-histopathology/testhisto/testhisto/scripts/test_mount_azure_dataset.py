#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import shutil
from time import time
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from health_azure.utils import get_authentication
from histopathology.scripts.mount_azure_dataset import mount_dataset


@pytest.mark.parametrize("dataset_id, nested_folder_path, id_column", [
    ('TCGA-CRCk', '', 'slide_id'),
    ('PANDA_tiles', 'PANDA_tiles_20210926-135446/panda_tiles_level1_224', 'tile_id'),
    ])
def test_all_images_exist(dataset_id, nested_folder_path, id_column, tmp_path_factory) -> None:
    """
    Test that the expected files are available when mounting datasets. Note that if the dataset
    contains a lot of images, the number of files that are checked will be restricted
    """
    dataset_path = "dataset.csv"
    mount_folder = tmp_path_factory.mktemp("datasets")
    mount_ctx, data_path = mount_dataset(dataset_id, mount_folder)
    mounted_data_path = Path(data_path)
    if nested_folder_path:
        mounted_data_path = mounted_data_path / nested_folder_path

    assert mounted_data_path.exists()
    assert mounted_data_path.is_dir()

    ## Check that the file 'dataset.csv' exists
    dataset_csv = mounted_data_path / dataset_path
    assert dataset_csv.is_file()
    start_time = time()
    data_df = pd.read_csv(dataset_csv, index_col=id_column)
    end_time = time()
    df_load_time = end_time - start_time
    logging.info(f"Time taken to load dataframe: {df_load_time}s")
    data_df_len = len(data_df)
    assert data_df_len > 0

    # Now check the images files exist. Note that the number of images that are checked is limited
    # as some datasets are too large e.g. PANDA_tiles contains c.683k images
    max_imgs_to_check = 3000
    image_names = data_df['image'].tolist()
    if data_df_len > max_imgs_to_check:
        import random
        image_names = random.sample(image_names, max_imgs_to_check)
    logging.info(f"Checking the existence of {min(data_df_len, max_imgs_to_check)} images")
    start_time = time()
    for image_name in image_names:
        expected_image_path = mounted_data_path / image_name
        assert expected_image_path.exists()

    end_time = time()
    img_check_time = end_time - start_time
    logging.info(f"Time taken to check that images exist: {img_check_time}")
    mount_ctx.stop()
    shutil.rmtree(mount_folder)
