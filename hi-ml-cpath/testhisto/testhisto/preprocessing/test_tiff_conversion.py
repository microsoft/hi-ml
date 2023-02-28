#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
import numpy as np
import pytest
from pathlib import Path
from typing import List, Dict

from health_cpath.preprocessing.tiff_conversion import ConvertWSIToTiffd, WSIFormat
from health_cpath.utils.naming import SlideKey


WSISamplesType = List[Dict[str, Path]]


@pytest.fixture
def wsi_samples(mock_panda_slides_root_dir) -> WSISamplesType:
    wsi_root_dir = mock_panda_slides_root_dir / "train_images"
    wsi_filenames = ["_1.tiff", "_2.tiff"]
    return [
        {
            SlideKey.SLIDE_ID: wsi_filename.split(".")[0],
            SlideKey.IMAGE: wsi_root_dir / wsi_filename,
        }
        for wsi_filename in wsi_filenames
    ]


@pytest.mark.parametrize("add_low_mag", [True, False])
def test_convert_wsi_to_tiff(add_low_mag: bool, wsi_samples: WSISamplesType, tmp_path: Path) -> None:
    """Test the conversion of the cyted dataset from ndpi to tiff."""
    target_mag = 2.5
    transform = ConvertWSIToTiffd(
        dest_dir=tmp_path,
        image_key=SlideKey.IMAGE,
        src_format=WSIFormat.TIFF,
        target_magnifications=[target_mag],
        add_lowest_magnification=add_low_mag,
        base_objective_power=target_mag,
        tile_size=16,
    )

    for sample in wsi_samples:
        transform(sample)

    original_files = [wsi_sample[SlideKey.IMAGE] for wsi_sample in wsi_samples]
    converted_files = [tmp_path / file.name for file in original_files]

    for converted_file, original_file in zip(converted_files, original_files):
        # check that the converted file exists and is not empty
        assert converted_file.exists()
        assert converted_file.stat().st_size > 0
        assert converted_file.name == original_file.name
        # get the original and converted wsi objects
        original_wsi = transform.wsi_reader.read(original_file)
        converted_wsi = transform.wsi_reader.read(converted_file)
        for o_level, c_level in zip(transform.get_target_levels(original_wsi), range(converted_wsi.level_count)):
            # For each level, check that the data is the same
            original_wsi_data = transform.get_level_data(original_wsi, o_level)
            converted_wsi_data = transform.get_level_data(converted_wsi, c_level)
            assert original_wsi_data.shape == converted_wsi_data.shape
            assert np.allclose(original_wsi_data, converted_wsi_data)
            # Check that the mpp is the same
            o_mpp = transform.wsi_reader.get_mpp(original_wsi, o_level)
            c_mpp = transform.wsi_reader.get_mpp(converted_wsi, c_level)
            assert all(map(lambda a, b: math.isclose(a, b, rel_tol=1e-3), o_mpp, c_mpp))
