#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Tuple, Union, Optional
from health_ml.utils.common_utils import is_gpu_available


from histopathology.datamodules.base_module import SlidesDataModule
from histopathology.datasets.base_dataset import SlidesDataset
from testhisto.utils.utils_testhisto import full_ml_test_data_path

no_gpu = not is_gpu_available()


class MockSlidesDataset(SlidesDataset):
    SLIDE_ID_COLUMN = "image_id"
    IMAGE_COLUMN = "image"
    LABEL_COLUMN = "isup_grade"

    METADATA_COLUMNS = ("data_provider", "isup_grade", "gleason_score")

    def __init__(
        self,
        root: Union[str, Path],
        dataset_csv: Optional[Union[str, Path]] = None,
        dataset_df: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__(root, dataset_csv, dataset_df, validate_columns=False)
        slide_ids = self.dataset_df.index
        self.dataset_df[self.IMAGE_COLUMN] = slide_ids + ".tiff"
        self.validate_columns()


class MockSlidesDataModule(SlidesDataModule):

    def get_splits(self) -> Tuple[MockSlidesDataset, MockSlidesDataset, MockSlidesDataset]:
        return (MockSlidesDataset(self.root_path), MockSlidesDataset(self.root_path), MockSlidesDataset(self.root_path))


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_tiling_on_the_fly() -> None:
    batch_size, tile_count, tile_size, level, channels = 1, 16, 28, 0, 3
    root_path = full_ml_test_data_path("whole_slide_images/pathmnist")
    datamodule = MockSlidesDataModule(
        root_path=root_path, batch_size=batch_size, tile_count=tile_count, tile_size=tile_size, level=level
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        # sanity check for expected shape
        tiles, wsi_id = sample["image"], sample["slide_id"][0]
        assert tiles.shape == (batch_size, tile_count, channels, tile_size, tile_size)

        # check tiling on the fly
        original_tile = np.load(root_path / f"{wsi_id}_tile.npy")
        for i in range(tile_count):
            assert (original_tile == tiles[0, i].numpy()).all()


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_tiling_without_fixed_tile_count() -> None:
    batch_size, tile_count, tile_size, level = 1, None, 28, 0
    min_expected_tile_count = 16
    root_path = full_ml_test_data_path("whole_slide_images/pathmnist")
    datamodule = MockSlidesDataModule(
        root_path=root_path, batch_size=batch_size, tile_count=tile_count, tile_size=tile_size, level=level
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        tiles = sample["image"]
        assert tiles.shape[1] >= min_expected_tile_count


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("level", [0, 1, 2])
def test_multi_resolution_tiling(level: int) -> None:
    batch_size, tile_count, channels = 1, 16, 3
    tile_size = 28 // 2 ** level
    root_path = full_ml_test_data_path("whole_slide_images/pathmnist")
    datamodule = MockSlidesDataModule(
        root_path=root_path, batch_size=batch_size, tile_count=tile_count, tile_size=tile_size, level=level
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        # sanity check for expected shape
        tiles, wsi_id = sample["image"], sample["slide_id"][0]
        assert tiles.shape == (batch_size, tile_count, channels, tile_size, tile_size)

        # check tiling on the fly at different resolutions
        original_tile = np.load(root_path / f"{wsi_id}_tile.npy")
        for i in range(tile_count):
            assert (original_tile[:, :: 2 ** level, :: 2 ** level] == tiles[0, i].numpy()).all()


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_overlapping_tiles() -> None:
    batch_size, level, tile_size, step = 1, 0, 28, 14
    root_path = full_ml_test_data_path("whole_slide_images/pathmnist")
    min_expected_tile_count, expected_tile_matches = 32, 16
    datamodule = MockSlidesDataModule(
        root_path=root_path, batch_size=batch_size, tile_size=tile_size, step=step, level=level
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        tiles, wsi_id = sample["image"], sample["slide_id"][0]
        assert tiles.shape[1] >= min_expected_tile_count

        original_tile = np.load(root_path / f"{wsi_id}_tile.npy")
        tile_matches = 0
        for tile in tiles[0]:
            tile_matches += int((tile.numpy() == original_tile).all())
        assert tile_matches == expected_tile_matches
