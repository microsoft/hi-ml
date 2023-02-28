#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import numpy as np
import pytest

from pathlib import Path
from typing import List, Dict
from unittest.mock import MagicMock
from health_cpath.datasets.panda_dataset import PandaDataset

from health_cpath.preprocessing.tiff_conversion import AMPERSAND, UNDERSCORE, ConvertWSIToTiffd, WSIFormat
from health_cpath.utils.naming import SlideKey
from health_cpath.utils.tiff_conversion_config import TiffConversionConfig


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


@pytest.mark.parametrize("replace_ampersand_by", [UNDERSCORE, ""])
@pytest.mark.parametrize("src_format", [WSIFormat.NDPI, WSIFormat.TIFF])
def test_get_tiff_path(src_format: WSIFormat, replace_ampersand_by: str) -> None:
    output_folder = Path("foo")
    transform = ConvertWSIToTiffd(
        output_folder=output_folder, image_key=SlideKey.IMAGE, src_format=src_format,
        replace_ampersand_by=replace_ampersand_by,
    )
    src_path = Path(f"root/h&e_foo.{src_format}")
    formated_path = transform.get_tiff_path(src_path)
    assert formated_path == output_folder / f"h{replace_ampersand_by}e_foo.tiff"


def test_base_objective_power(wsi_samples: WSISamplesType) -> None:
    target_mag = 2.5
    transform = ConvertWSIToTiffd(
        output_folder=Path("foo"),
        image_key=SlideKey.IMAGE,
        src_format=WSIFormat.TIFF,
        target_magnifications=[target_mag],
        base_objective_power=None,
    )
    wsi_obj = transform.wsi_reader.read(wsi_samples[0][SlideKey.IMAGE])
    with pytest.raises(ValueError, match=r"Could not find openslide.objective-power in"):
        _ = transform._get_base_objective_power(wsi_obj)

    # openslide.objective-power is not part of the wsi object properties but base_objective_power is set
    transform.base_objective_power = target_mag
    base_obj_power = transform._get_base_objective_power(wsi_obj)
    assert base_obj_power == target_mag

    # openslide.objective-power is part of the wsi object properties
    mock_wsi_obj = MagicMock(properties={transform.OBJECTIVE_POWER_KEY: 10})
    base_obj_power = transform._get_base_objective_power(mock_wsi_obj)
    assert base_obj_power == 10


def test_get_taget_levels(wsi_samples: WSISamplesType) -> None:
    target_mag = 2.5
    transform = ConvertWSIToTiffd(
        output_folder=Path("foo"),
        image_key=SlideKey.IMAGE,
        src_format=WSIFormat.TIFF,
        target_magnifications=[target_mag],
        base_objective_power=target_mag,
    )
    wsi_obj = transform.wsi_reader.read(wsi_samples[0][SlideKey.IMAGE])

    target_levels = transform.get_target_levels(wsi_obj)
    assert len(target_levels) == 1
    assert target_levels[0] == 0

    # add the lowest magnification, the wsi had 3 levels
    transform.add_lowest_magnification = True
    target_levels = transform.get_target_levels(wsi_obj)
    assert len(target_levels) == 2
    assert target_levels == [0, 2]
    transform.add_lowest_magnification = False

    # set the base objective power to 5x
    transform.base_objective_power = target_mag * 2
    target_levels = transform.get_target_levels(wsi_obj)
    assert len(target_levels) == 1
    assert target_levels[0] == 1


def test_get_options(wsi_samples: WSISamplesType) -> None:
    transform = ConvertWSIToTiffd(output_folder=Path("foo"), tile_size=16)
    assert transform.RESOLUTION_UNIT == "centimeter"

    # wrong resolution unit
    mock_wsi_obj = MagicMock(properties={transform.RESOLUTION_UNIT_KEY: "micrometer"})
    with pytest.raises(ValueError, match=r"Resolution unit is not in centimeter"):
        _ = transform.get_options(mock_wsi_obj)
    # correct resolution unit in centimeter
    wsi_obj = transform.wsi_reader.read(wsi_samples[0][SlideKey.IMAGE])
    options = transform.get_options(wsi_obj)
    assert options["resolutionunit"] == transform.RESOLUTION_UNIT
    assert options["software"] == transform.SOFTWARE


def validate_tiff_conversion(
    converted_files: List[Path],
    original_files: List[Path],
    transform: ConvertWSIToTiffd,
    same_format: bool = True,
    subfolder: str = "",
) -> None:
    """Validate the conversion of a list of files to tiff."""
    for converted_file, original_file in zip(converted_files, original_files):
        # check that the converted file exists and is not empty
        assert converted_file.exists()
        assert converted_file.stat().st_size > 0
        assert converted_file.suffix == ".tiff"
        assert converted_file.parent == transform.output_folder / subfolder
        assert converted_file.name == original_file.name if same_format else converted_file.name != original_file.name
        assert converted_file.stem == original_file.stem.replace(AMPERSAND, transform.replace_ampersand_by)
        # get the original and converted wsi objects
        original_wsi = transform.wsi_reader.read(original_file)
        converted_wsi = transform.wsi_reader.read(converted_file)
        # check that the number of levels is the same
        target_levels = transform.get_target_levels(original_wsi)
        assert len(target_levels) == converted_wsi.level_count
        # check that the data and mpp are the same for each level
        for original_level, converted_level in zip(target_levels, range(converted_wsi.level_count)):
            # For each level, check that the data is the same
            original_wsi_data = transform.get_level_data(original_wsi, original_level)
            converted_wsi_data = transform.get_level_data(converted_wsi, converted_level)
            assert original_wsi_data.shape == converted_wsi_data.shape
            assert np.allclose(original_wsi_data, converted_wsi_data)
            # Check that the mpp is the same
            o_mpp = transform.wsi_reader.get_mpp(original_wsi, original_level)
            c_mpp = transform.wsi_reader.get_mpp(converted_wsi, converted_level)
            assert all(map(lambda a, b: math.isclose(a, b, rel_tol=1e-3), o_mpp, c_mpp))


@pytest.mark.parametrize("add_low_mag", [True, False])
def test_convert_wsi_to_tiff(add_low_mag: bool, wsi_samples: WSISamplesType, tmp_path: Path) -> None:
    """Test the conversion of the cyted dataset from tiff to tiff."""
    target_mag = 2.5
    transform = ConvertWSIToTiffd(
        output_folder=tmp_path,
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

    validate_tiff_conversion(converted_files, original_files, transform)


def test_tiff_conversion_config(mock_panda_slides_root_dir: Path, tmp_path: Path) -> None:
    dataset = PandaDataset(mock_panda_slides_root_dir)
    target_mag = 5
    limit = 2
    conversion_config = TiffConversionConfig(
        src_format=WSIFormat.TIFF,
        target_magnifications=[target_mag / 2],
        base_objective_power=target_mag,
        tile_size=16,
        num_workers=1,
    )
    dataset.dataset_df = dataset.dataset_df.iloc[:limit]
    conversion_config.run(dataset, tmp_path, wsi_subfolder="train_images")

    # Test the original dataset is not modified
    original_dataset = PandaDataset(mock_panda_slides_root_dir)
    assert original_dataset.dataset_df.iloc[:limit].equals(dataset.dataset_df)

    # Validate that the new dataset has the expected number of samples
    converted_dataset = PandaDataset(tmp_path)
    assert converted_dataset.dataset_df.shape[0] == dataset.dataset_df.shape[0]
    assert converted_dataset.dataset_df.index.equals(dataset.dataset_df.index)

    original_files = [Path(dataset[i][SlideKey.IMAGE]) for i in range(limit)]
    converted_files = [Path(converted_dataset[i][SlideKey.IMAGE]) for i in range(limit)]
    transform = conversion_config.get_transform(tmp_path)
    validate_tiff_conversion(converted_files, original_files, transform, same_format=True, subfolder="train_images")
