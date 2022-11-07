#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
import torch
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from health_cpath.datamodules.base_module import HistoDataModule
from health_cpath.datamodules.panda_module import PandaSlidesDataModule, PandaTilesDataModule
from health_cpath.utils.naming import ModelKey, SlideKey
from health_ml.utils.common_utils import is_gpu_available
from testhisto.utils.utils_testhisto import run_distributed


no_gpu = not is_gpu_available()


def _assert_correct_bag_sizes(datamodule: HistoDataModule, max_bag_size: int, max_bag_size_inf: int,
                              max_expected_bag_sizes: List[int]) -> None:
    for model_key, bag_size in zip([m for m in ModelKey], [max_bag_size, max_bag_size_inf, max_bag_size_inf]):
        assert datamodule.bag_sizes[model_key] == bag_size

    def _assert_bag_size_matching(dataloader: DataLoader, expected_bag_sizes: List[int]) -> None:
        sample = next(iter(dataloader))
        for i, slide in enumerate(sample[SlideKey.IMAGE]):
            assert slide.shape[0] == expected_bag_sizes[i]

    _assert_bag_size_matching(datamodule.train_dataloader(), [max_bag_size, max_bag_size])
    expected_bag_sizes = max_expected_bag_sizes if not max_bag_size_inf else [max_bag_size_inf, max_bag_size_inf]
    _assert_bag_size_matching(datamodule.val_dataloader(), expected_bag_sizes)
    _assert_bag_size_matching(datamodule.test_dataloader(), expected_bag_sizes)


def _assert_correct_batch_sizes(datamodule: HistoDataModule, batch_size: int, batch_size_inf: Optional[int]) -> None:
    batch_size_inf = batch_size_inf if batch_size_inf is not None else batch_size
    for model_key, _batch_size in zip([m for m in ModelKey], [batch_size, batch_size_inf, batch_size_inf]):
        assert datamodule.batch_sizes[model_key] == _batch_size

    def _assert_batch_size_matching(dataloader: DataLoader, expected_batch_size: int) -> None:
        sample = next(iter(dataloader))
        assert len(sample[SlideKey.IMAGE]) == expected_batch_size

    _assert_batch_size_matching(datamodule.train_dataloader(), batch_size)
    _assert_batch_size_matching(datamodule.val_dataloader(), batch_size_inf)
    _assert_batch_size_matching(datamodule.test_dataloader(), batch_size_inf)


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("max_bag_size, max_bag_size_inf", [(2, 0), (2, 3)])
def test_slides_datamodule_different_bag_sizes(
    mock_panda_slides_root_dir: Path, max_bag_size: int, max_bag_size_inf: int
) -> None:
    datamodule = PandaSlidesDataModule(
        root_path=mock_panda_slides_root_dir,
        batch_size=2,
        max_bag_size=max_bag_size,
        max_bag_size_inf=max_bag_size_inf,
        tile_size=28,
        level=0,
    )
    # To account for the fact that slides datamodule fomats 0 to None so that it's compatible with TileOnGrid transform
    max_bag_size_inf = max_bag_size_inf if max_bag_size_inf != 0 else None  # type: ignore
    _assert_correct_bag_sizes(datamodule, max_bag_size, max_bag_size_inf, [4, 4])


@pytest.mark.parametrize("max_bag_size, max_bag_size_inf", [(2, 0), (2, 3)])
def test_tiles_datamodule_different_bag_sizes(
    mock_panda_tiles_root_dir: Path, max_bag_size: int, max_bag_size_inf: int
) -> None:
    datamodule = PandaTilesDataModule(
        root_path=mock_panda_tiles_root_dir,
        batch_size=2,
        max_bag_size=max_bag_size,
        max_bag_size_inf=max_bag_size_inf,
    )
    _assert_correct_bag_sizes(datamodule, max_bag_size, max_bag_size_inf, max_expected_bag_sizes=[4, 5])


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("batch_size, batch_size_inf", [(2, 2), (2, 1), (2, None)])
def test_slides_datamodule_different_batch_sizes(
    mock_panda_slides_root_dir: Path, batch_size: int, batch_size_inf: Optional[int],
) -> None:
    datamodule = PandaSlidesDataModule(
        root_path=mock_panda_slides_root_dir,
        batch_size=batch_size,
        batch_size_inf=batch_size_inf,
        max_bag_size=16,
        max_bag_size_inf=16,
        tile_size=28,
        level=0,
    )
    _assert_correct_batch_sizes(datamodule, batch_size, batch_size_inf)


@pytest.mark.parametrize("batch_size, batch_size_inf", [(2, 2), (2, 1), (2, None)])
def test_tiles_datamodule_different_batch_sizes(
    mock_panda_tiles_root_dir: Path, batch_size: int, batch_size_inf: Optional[int],
) -> None:
    datamodule = PandaTilesDataModule(
        root_path=mock_panda_tiles_root_dir,
        batch_size=batch_size,
        batch_size_inf=batch_size_inf,
        max_bag_size=16,
        max_bag_size_inf=16,
    )
    _assert_correct_batch_sizes(datamodule, batch_size, batch_size_inf)


def _assert_sampler_state(datamodule: HistoDataModule, stages: List[ModelKey], expected_none: bool) -> None:
    sampler_instances = {
        ModelKey.TRAIN: RandomSampler if expected_none else DistributedSampler,
        ModelKey.VAL: SequentialSampler,
        ModelKey.TEST: SequentialSampler,
    }
    for stage in stages:
        datamodule_sampler = datamodule._get_ddp_sampler(getattr(datamodule, f'{stage}_dataset'), stage)
        assert (datamodule_sampler is None) == expected_none
        dataloader = getattr(datamodule, f'{stage.value}_dataloader')()
        assert isinstance(dataloader.sampler, sampler_instances[stage])


def _test_datamodule_pl_ddp_sampler_true(
    datamodule: HistoDataModule, rank: int = 0, world_size: int = 1, device: str = "cpu"
) -> None:
    datamodule.setup()
    _assert_sampler_state(datamodule, [ModelKey.TRAIN, ModelKey.VAL, ModelKey.TEST], expected_none=True)


def _test_datamodule_pl_ddp_sampler_false(
    datamodule: HistoDataModule, rank: int = 0, world_size: int = 1, device: str = "cpu"
) -> None:
    datamodule.setup()
    _assert_sampler_state(datamodule, [ModelKey.VAL, ModelKey.TEST], expected_none=True)
    _assert_sampler_state(datamodule, [ModelKey.TRAIN], expected_none=False)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_slides_datamodule_pl_replace_sampler_ddp(mock_panda_slides_root_dir: Path) -> None:
    slides_datamodule = PandaSlidesDataModule(root_path=mock_panda_slides_root_dir,
                                              pl_replace_sampler_ddp=True,
                                              seed=42)
    run_distributed(_test_datamodule_pl_ddp_sampler_true, [slides_datamodule], world_size=2)
    slides_datamodule.pl_replace_sampler_ddp = False
    run_distributed(_test_datamodule_pl_ddp_sampler_false, [slides_datamodule], world_size=2)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_tiles_datamodule_pl_replace_sampler_ddp(mock_panda_tiles_root_dir: Path) -> None:
    tiles_datamodule = PandaTilesDataModule(root_path=mock_panda_tiles_root_dir, seed=42, pl_replace_sampler_ddp=True)
    run_distributed(_test_datamodule_pl_ddp_sampler_true, [tiles_datamodule], world_size=2)
    tiles_datamodule.pl_replace_sampler_ddp = False
    run_distributed(_test_datamodule_pl_ddp_sampler_false, [tiles_datamodule], world_size=2)


def test_assertion_error_missing_seed(mock_panda_slides_root_dir: Path) -> None:
    with pytest.raises(AssertionError, match="seed must be set when using distributed training for reproducibility"):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=2):
                slides_datamodule = PandaSlidesDataModule(
                    root_path=mock_panda_slides_root_dir, pl_replace_sampler_ddp=False
                )
                slides_datamodule._get_ddp_sampler(MagicMock(), ModelKey.TRAIN)
