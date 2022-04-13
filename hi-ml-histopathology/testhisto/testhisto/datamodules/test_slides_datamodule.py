#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
import numpy as np
from pathlib import Path

from health_ml.utils.common_utils import is_gpu_available

from histopathology.utils.naming import SlideKey
from histopathology.datamodules.panda_module import PandaSlidesDataModule
from testhisto.mocks.slides_generator import (
    MockPandaSlidesGenerator,
    MockHistoDataType,
)


no_gpu = not is_gpu_available()


@pytest.fixture(scope="session")
def mock_panda_slides_root_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_root_dir = tmp_path_factory.mktemp("mock_wsi")
    wsi_generator = MockPandaSlidesGenerator(
        tmp_path=tmp_root_dir,
        mock_type=MockHistoDataType.PATHMNIST,
        n_tiles=1,
        n_slides=10,
        n_repeat_diag=4,
        n_repeat_tile=2,
        n_channels=3,
        n_levels=3,
        tile_size=28,
        background_val=255,
    )
    wsi_generator.generate_mock_histo_data()
    return tmp_root_dir


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_tiling_on_the_fly(mock_panda_slides_root_dir: Path) -> None:
    batch_size = 1
    tile_count = 16
    tile_size = 28
    level = 0
    channels = 3
    datamodule = PandaSlidesDataModule(
        root_path=mock_panda_slides_root_dir,
        batch_size=batch_size,
        tile_count=tile_count,
        tile_size=tile_size,
        level=level,
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        # sanity check for expected shape
        tiles, wsi_id = sample[SlideKey.IMAGE], sample[SlideKey.SLIDE_ID][0]
        assert tiles.shape == (batch_size, tile_count, channels, tile_size, tile_size)

        # check tiling on the fly
        original_tile = np.load(mock_panda_slides_root_dir / "dump_tiles" / f"{wsi_id}.npy")[0]
        for i in range(tile_count):
            assert (original_tile == tiles[0, i].numpy()).all()


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_tiling_without_fixed_tile_count(mock_panda_slides_root_dir: Path) -> None:
    batch_size = 1
    tile_count = None
    tile_size = 28
    level = 0
    min_expected_tile_count = 16
    datamodule = PandaSlidesDataModule(
        root_path=mock_panda_slides_root_dir,
        batch_size=batch_size,
        tile_count=tile_count,
        tile_size=tile_size,
        level=level,
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        tiles = sample[SlideKey.IMAGE]
        assert tiles.shape[1] >= min_expected_tile_count


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("level", [0, 1, 2])
def test_multi_resolution_tiling(level: int, mock_panda_slides_root_dir: Path) -> None:
    batch_size = 1
    tile_count = 16
    channels = 3
    tile_size = 28 // 2 ** level
    datamodule = PandaSlidesDataModule(
        root_path=mock_panda_slides_root_dir,
        batch_size=batch_size,
        tile_count=tile_count,
        tile_size=tile_size,
        level=level,
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        # sanity check for expected shape
        tiles, wsi_id = sample[SlideKey.IMAGE], sample[SlideKey.SLIDE_ID][0]
        assert tiles.shape == (batch_size, tile_count, channels, tile_size, tile_size)

        # check tiling on the fly at different resolutions
        original_tile = np.load(mock_panda_slides_root_dir / "dump_tiles" / f"{wsi_id}.npy")[0]
        for i in range(tile_count):
            # multi resolution mock data has been created via 2 factor downsampling
            assert (original_tile[:, :: 2 ** level, :: 2 ** level] == tiles[0, i].numpy()).all()


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_overlapping_tiles(mock_panda_slides_root_dir: Path) -> None:
    batch_size = 1
    tile_size = 28
    level = 0
    step = 14
    expected_tile_matches = 16
    min_expected_tile_count = 32
    datamodule = PandaSlidesDataModule(
        root_path=mock_panda_slides_root_dir, batch_size=batch_size, tile_size=tile_size, step=step, level=level
    )
    dataloader = datamodule.train_dataloader()
    for sample in dataloader:
        tiles, wsi_id = sample[SlideKey.IMAGE], sample[SlideKey.SLIDE_ID][0]
        assert tiles.shape[1] >= min_expected_tile_count

        original_tile = np.load(mock_panda_slides_root_dir / "dump_tiles" / f"{wsi_id}.npy")[0]
        tile_matches = 0
        for i, tile in enumerate(tiles[0]):
            tile_matches += int((tile.numpy() == original_tile).all())
        assert tile_matches == expected_tile_matches
