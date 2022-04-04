#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import py
import pytest
import numpy as np

from health_ml.utils.common_utils import is_gpu_available
from testhisto.utils.utils_slides_datamodule import (
    MockSlidesDataModule,
    MockWSIGenerator,
    MockWSIType,
    MockSlidesDataset,
)


no_gpu = not is_gpu_available()
N_REPEAT_DIAG = 4


@pytest.fixture(scope="session")
def mock_wsi_root_dir(tmpdir_factory: pytest.TempdirFactory) -> py.path.local:
    tmp_root_dir = tmpdir_factory.mktemp("mock_wsi")
    wsi_generator = MockWSIGenerator(
        tmp_path=tmp_root_dir,
        mock_type=MockWSIType.PATHMNIST,
        batch_size=1,
        n_samples=4,
        n_repeat_diag=N_REPEAT_DIAG,
        n_repeat_tile=2,
        n_channels=3,
        n_levels=3,
        tile_size=28,
        background_val=255,
    )
    wsi_generator.generate_mock_wsi()
    return tmp_root_dir


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_tiling_on_the_fly(mock_wsi_root_dir: py.path.local) -> None:
    batch_size = 1
    tile_count = 16
    tile_size = 28
    level = 0
    channels = 3
    datamodule = MockSlidesDataModule(
        root_path=mock_wsi_root_dir, batch_size=batch_size, tile_count=tile_count, tile_size=tile_size, level=level
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        # sanity check for expected shape
        tiles, wsi_id = sample[MockSlidesDataset.IMAGE_COLUMN], sample[MockSlidesDataset.SLIDE_ID_COLUMN][0]
        assert tiles.shape == (batch_size, tile_count, channels, tile_size, tile_size)

        # check tiling on the fly
        original_tile = np.load(str(mock_wsi_root_dir.join(f"{wsi_id}_tile.npy")))[0]
        for i in range(tile_count):
            assert (original_tile == tiles[0, i].numpy()).all()


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_tiling_without_fixed_tile_count(mock_wsi_root_dir: py.path.local) -> None:
    batch_size = 1
    tile_count = None
    tile_size = 28
    level = 0
    min_expected_tile_count = 16
    datamodule = MockSlidesDataModule(
        root_path=mock_wsi_root_dir, batch_size=batch_size, tile_count=tile_count, tile_size=tile_size, level=level
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        tiles = sample[MockSlidesDataset.IMAGE_COLUMN]
        assert tiles.shape[1] >= min_expected_tile_count


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("level", [0, 1, 2])
def test_multi_resolution_tiling(level: int, mock_wsi_root_dir: py.path.local) -> None:
    batch_size = 1
    tile_count = 16
    channels = 3
    tile_size = 28 // 2 ** level
    datamodule = MockSlidesDataModule(
        root_path=mock_wsi_root_dir, batch_size=batch_size, tile_count=tile_count, tile_size=tile_size, level=level
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        # sanity check for expected shape
        tiles, wsi_id = sample[MockSlidesDataset.IMAGE_COLUMN], sample[MockSlidesDataset.SLIDE_ID_COLUMN][0]
        assert tiles.shape == (batch_size, tile_count, channels, tile_size, tile_size)

        # check tiling on the fly at different resolutions
        original_tile = np.load(str(mock_wsi_root_dir.join(f"{wsi_id}_tile.npy")))[0]
        for i in range(tile_count):
            # multi resolution mock data has been created via 2 factor downsampling
            assert (original_tile[:, :: 2 ** level, :: 2 ** level] == tiles[0, i].numpy()).all()


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_overlapping_tiles(mock_wsi_root_dir: py.path.local) -> None:
    batch_size = 1
    tile_size = 28
    level = 0
    step = 14
    expected_tile_matches = 16
    min_expected_tile_count = 32
    datamodule = MockSlidesDataModule(
        root_path=mock_wsi_root_dir, batch_size=batch_size, tile_size=tile_size, step=step, level=level
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        tiles, wsi_id = sample[MockSlidesDataset.IMAGE_COLUMN], sample[MockSlidesDataset.SLIDE_ID_COLUMN][0]
        assert tiles.shape[1] >= min_expected_tile_count

        original_tile = np.load(str(mock_wsi_root_dir.join(f"{wsi_id}_tile.npy")))[0]
        tile_matches = 0
        for i, tile in enumerate(tiles[0]):
            tile_matches += int((tile.numpy() == original_tile).all())
        assert tile_matches == expected_tile_matches
