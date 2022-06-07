

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import shutil
from typing import Generator, Dict, Callable, Union, Tuple
import pytest
import logging
import numpy as np
import torch
from pathlib import Path
from monai.transforms import RandFlipd

from health_ml.utils.common_utils import is_gpu_available
from histopathology.datamodules.base_module import SlidesDataModule
from histopathology.datasets.panda_dataset import PandaDataset

from histopathology.utils.naming import SlideKey, ModelKey
from histopathology.datamodules.panda_module import PandaSlidesDataModule
from testhisto.mocks.slides_generator import (
    MockPandaSlidesGenerator,
    MockHistoDataType,
    TilesPositioningType,
)

no_gpu = not is_gpu_available()


@pytest.fixture(scope="session")
def mock_panda_slides_root_dir(
    tmp_path_factory: pytest.TempPathFactory, tmp_path_to_pathmnist_dataset: Path
) -> Generator:
    tmp_root_dir = tmp_path_factory.mktemp("mock_wsi")
    wsi_generator = MockPandaSlidesGenerator(
        dest_data_path=tmp_root_dir,
        src_data_path=tmp_path_to_pathmnist_dataset,
        mock_type=MockHistoDataType.PATHMNIST,
        n_tiles=1,
        n_slides=10,
        n_repeat_diag=4,
        n_repeat_tile=2,
        n_channels=3,
        n_levels=3,
        tile_size=28,
        background_val=255,
        tiles_pos_type=TilesPositioningType.DIAGONAL,
    )
    logging.info("Generating mock whole slide images that will be deleted at the end of the session.")
    wsi_generator.generate_mock_histo_data()
    yield tmp_root_dir
    shutil.rmtree(tmp_root_dir)


@pytest.fixture(scope="session")
def mock_panda_slides_root_with_different_n_tiles(
    tmp_path_factory: pytest.TempPathFactory, tmp_path_to_pathmnist_dataset: Path
) -> Generator:
    tmp_root_dir = tmp_path_factory.mktemp("mock_slides_diff_n_tiles")
    wsi_generator = MockPandaSlidesGenerator(
        dest_data_path=tmp_root_dir,
        src_data_path=tmp_path_to_pathmnist_dataset,
        mock_type=MockHistoDataType.PATHMNIST,
        n_tiles=4,
        n_slides=6,
        n_channels=3,
        n_levels=3,
        tile_size=28,
        background_val=255,
        tiles_pos_type=TilesPositioningType.RANDOM,
        n_tiles_list=[4, 5, 6, 7, 8, 9],
    )
    logging.info("Generating temporary mock slides that will be deleted at the end of the session.")
    wsi_generator.generate_mock_histo_data()
    yield tmp_root_dir
    shutil.rmtree(tmp_root_dir)


def get_original_tile(mock_dir: Path, wsi_id: str) -> np.ndarray:
    return np.load(mock_dir / "dump_tiles" / f"{wsi_id}.npy")[0]


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_tiling_on_the_fly(mock_panda_slides_root_dir: Path) -> None:
    batch_size = 1
    tile_count = 16
    tile_size = 28
    level = 0
    channels = 3
    assert_batch_index = 0
    datamodule = PandaSlidesDataModule(
        root_path=mock_panda_slides_root_dir,
        batch_size=batch_size,
        max_bag_size=tile_count,
        tile_size=tile_size,
        level=level,
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        # sanity check for expected shape
        tiles, wsi_id = sample[SlideKey.IMAGE], sample[SlideKey.SLIDE_ID][assert_batch_index]
        assert len(tiles) == batch_size
        assert tiles[assert_batch_index].shape == (tile_count, channels, tile_size, tile_size)

        # check tiling on the fly
        original_tile = get_original_tile(mock_panda_slides_root_dir, wsi_id)
        for i in range(tile_count):
            assert (original_tile == tiles[assert_batch_index][i].numpy()).all()


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_tiling_without_fixed_tile_count(mock_panda_slides_root_dir: Path) -> None:
    batch_size = 1
    tile_count = None
    tile_size = 28
    level = 0
    assert_batch_index = 0
    min_expected_tile_count = 16
    datamodule = PandaSlidesDataModule(
        root_path=mock_panda_slides_root_dir,
        batch_size=batch_size,
        max_bag_size=tile_count,
        tile_size=tile_size,
        level=level,
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        tiles = sample[SlideKey.IMAGE]
        assert tiles[assert_batch_index].shape[0] >= min_expected_tile_count


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("level", [0, 1, 2])
def test_multi_resolution_tiling(level: int, mock_panda_slides_root_dir: Path) -> None:
    batch_size = 1
    tile_count = 16
    channels = 3
    tile_size = 28 // 2 ** level
    assert_batch_index = 0
    datamodule = PandaSlidesDataModule(
        root_path=mock_panda_slides_root_dir,
        batch_size=batch_size,
        max_bag_size=tile_count,
        tile_size=tile_size,
        level=level,
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        # sanity check for expected shape
        tiles, wsi_id = sample[SlideKey.IMAGE], sample[SlideKey.SLIDE_ID][assert_batch_index]
        assert tiles[assert_batch_index].shape == (tile_count, channels, tile_size, tile_size)

        # check tiling on the fly at different resolutions
        original_tile = get_original_tile(mock_panda_slides_root_dir, wsi_id)
        for i in range(tile_count):
            # multi resolution mock data has been created via 2 factor downsampling
            assert (original_tile[:, :: 2 ** level, :: 2 ** level] == tiles[assert_batch_index][i].numpy()).all()


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 2])
def test_overlapping_tiles(batch_size: int, mock_panda_slides_root_dir: Path) -> None:
    tile_size = 28
    level = 0
    step = 14
    expected_tile_matches = 16
    min_expected_tile_count = 32
    assert_batch_index = 0
    datamodule = PandaSlidesDataModule(
        root_path=mock_panda_slides_root_dir,
        max_bag_size=None,
        batch_size=batch_size,
        tile_size=tile_size,
        step=step,
        level=level
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        tiles, wsi_id = sample[SlideKey.IMAGE], sample[SlideKey.SLIDE_ID][assert_batch_index]
        assert tiles[assert_batch_index].shape[0] >= min_expected_tile_count

        original_tile = get_original_tile(mock_panda_slides_root_dir, wsi_id)
        tile_matches = 0
        for _, tile in enumerate(tiles[assert_batch_index]):
            tile_matches += int((tile.numpy() == original_tile).all())
        assert tile_matches == expected_tile_matches


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_train_test_transforms(mock_panda_slides_root_dir: Path) -> None:
    def get_transforms_dict() -> Dict[ModelKey, Union[Callable, None]]:
        train_transform = RandFlipd(keys=[SlideKey.IMAGE], spatial_axis=0, prob=1.0)
        return {ModelKey.TRAIN: train_transform, ModelKey.VAL: None, ModelKey.TEST: None}   # type: ignore

    def retrieve_tiles(dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        tiles_dict = {}
        assert_batch_index = 0
        for sample in dataloader:
            tiles, wsi_id = sample[SlideKey.IMAGE], sample[SlideKey.SLIDE_ID][assert_batch_index]
            tiles_dict.update({wsi_id: tiles[assert_batch_index]})
        return tiles_dict

    batch_size = 1
    tile_count = 4
    tile_size = 28
    level = 0
    flipdatamodule = PandaSlidesDataModule(
        root_path=mock_panda_slides_root_dir,
        batch_size=batch_size,
        max_bag_size=tile_count,
        max_bag_size_inf=0,
        tile_size=tile_size,
        level=level,
        transforms_dict=get_transforms_dict(),
    )
    flip_train_tiles = retrieve_tiles(flipdatamodule.train_dataloader())
    flip_val_tiles = retrieve_tiles(flipdatamodule.val_dataloader())
    flip_test_tiles = retrieve_tiles(flipdatamodule.test_dataloader())

    for wsi_id in flip_train_tiles.keys():
        original_tile = get_original_tile(mock_panda_slides_root_dir, wsi_id)
        # the first dimension is the channel, flipping happened on the horizontal axis of the image
        transformed_original_tile = np.flip(original_tile, axis=1)
        for tile in flip_train_tiles[wsi_id]:
            assert (tile.numpy() == transformed_original_tile).all()

    for wsi_id in flip_val_tiles.keys():
        original_tile = get_original_tile(mock_panda_slides_root_dir, wsi_id)
        for tile in flip_val_tiles[wsi_id]:
            # no transformation has been applied to val tiles
            assert (tile.numpy() == original_tile).all()

    for wsi_id in flip_test_tiles.keys():
        original_tile = get_original_tile(mock_panda_slides_root_dir, wsi_id)
        for tile in flip_test_tiles[wsi_id]:
            # no transformation has been applied to test tiles
            assert (tile.numpy() == original_tile).all()


class MockPandaSlidesDataModule(SlidesDataModule):
    """ MockPandaSlidesDataModule is the child class of SlidesDataModule specific to PANDA dataset
    Method get_splits() returns the train, val, test splits from the PANDA dataset. Here we return the same subsets for
    train/val and test to make sure train_dataloader returns a fixed n_tiles and test and validation dataloaders return
    all available tiles in the whole slide image for whole slide inference.
    """

    def get_splits(self) -> Tuple[PandaDataset, PandaDataset, PandaDataset]:

        return (PandaDataset(self.root_path), PandaDataset(self.root_path), PandaDataset(self.root_path))


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 2])
def test_whole_slide_inference(batch_size: int, mock_panda_slides_root_with_different_n_tiles: Path) -> None:
    tile_count = 2
    tile_size = 28
    level = 0
    assert_batch_index = 0
    n_tiles_list = [4, 5, 6, 7, 8, 9]

    datamodule = MockPandaSlidesDataModule(
        root_path=mock_panda_slides_root_with_different_n_tiles,
        batch_size=batch_size,
        max_bag_size=tile_count,
        max_bag_size_inf=0,
        tile_size=tile_size,
        level=level,
    )
    train_dataloader = datamodule.train_dataloader()
    for sample in train_dataloader:
        tiles = sample[SlideKey.IMAGE]
        assert tiles[assert_batch_index].shape[0] == tile_count

    def assert_whole_slide_inference_with_all_tiles(dataloader: torch.utils.data.DataLoader) -> None:
        for i, sample in enumerate(dataloader):
            tiles = sample[SlideKey.IMAGE]
            assert tiles[assert_batch_index].shape[0] == n_tiles_list[i * batch_size]

    assert_whole_slide_inference_with_all_tiles(datamodule.val_dataloader())
    assert_whole_slide_inference_with_all_tiles(datamodule.test_dataloader())
