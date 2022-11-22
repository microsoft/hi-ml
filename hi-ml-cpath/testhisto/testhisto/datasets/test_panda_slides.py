#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, List
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything

from health_cpath.configs.classification.DeepSMILESlidesPandaBenchmark import DeepSMILESlidesPandaBenchmark
from health_cpath.datasets.panda_dataset import PandaDataset
from health_cpath.preprocessing.loading import ROIType, WSIBackend
from health_cpath.utils.naming import SlideKey
from testhisto.mocks.base_data_generator import MockHistoDataType
from testhisto.mocks.slides_generator import MockPandaSlidesGenerator, TilesPositioningType

try:
    from cucim import CuImage  # noqa: F401
    has_cucim = True
except:  # noqa: E722
    has_cucim = False


@pytest.mark.skipif(not has_cucim, reason="Test requires CUCIM library")
@pytest.mark.gpu
def test_panda_reproducibility(tmp_path: Path) -> None:
    """Check if subsequent enumerations of the Panda dataset produce identical sequences of tiles."""
    seed_everything(seed=123, workers=True)
    tile_size = 28
    num_tiles = 4
    wsi_generator = MockPandaSlidesGenerator(
        dest_data_path=tmp_path,
        mock_type=MockHistoDataType.FAKE,
        n_tiles=num_tiles,
        n_slides=10,
        n_channels=3,
        n_levels=3,
        tile_size=tile_size,
        background_val=255,
        tiles_pos_type=TilesPositioningType.RANDOM
    )
    wsi_generator.generate_mock_histo_data()

    container = DeepSMILESlidesPandaBenchmark()
    container.tile_size = tile_size
    container.max_bag_size = num_tiles
    container.local_datasets = [tmp_path]
    container.backend = WSIBackend.CUCIM
    container.roi_type = ROIType.FOREGROUND
    container.margin = 0
    container.level = 0

    def test_data_items_are_equal(loader_fn_names: List[str]) -> None:
        """Creates a new data module from the container, and checks if all the data loaders specified in
        `loader_fn_names` return the same set of items when enumerated twice."""
        data_module = container.get_data_module()
        data_module.dataloader_kwargs = {**data_module.dataloader_kwargs,
                                         "multiprocessing_context": None,
                                         "num_workers": 0}

        def get_item_from_new_loader(loader_fn_name: str) -> Any:
            """Get a new dataloader with the given name (for example, 'train_dataloader') and returns the first item."""
            loader_fn = getattr(data_module, loader_fn_name)
            loader = loader_fn()
            iterator = iter(loader)
            return next(iterator)

        for loader_fn_name in loader_fn_names:
            print(f"Checking {loader_fn_name}")
            item1 = get_item_from_new_loader(loader_fn_name)
            item2 = get_item_from_new_loader(loader_fn_name)
            assert item1[SlideKey.SLIDE_ID] == item2[SlideKey.SLIDE_ID], "Order of slides must match"
            assert len(item1[SlideKey.IMAGE]) == len(item2[SlideKey.IMAGE]), "Length of images must match"
            for i in range(len(item1[SlideKey.IMAGE])):
                image1 = item1[SlideKey.IMAGE][i]
                image2 = item2[SlideKey.IMAGE][i]
                if len(np.unique(image1)) == 1:
                    assert False, "Something is wrong here, image1 only has a single value"
                if len(np.unique(image2)) == 1:
                    assert False, "Something is wrong here, image2 only has a single value"
                assert torch.allclose(image1, image2), "Images don't match"

    # If no fixed random seed is set, the data items should not match in subsequent calls to the loader,
    # and hence raise an AssertionError
    with mock.patch.object(container, "get_effective_random_seed", MagicMock(return_value=None)):
        with pytest.raises(AssertionError):
            test_data_items_are_equal(["train_dataloader"])

    # When using a fixed see, all 3 dataloaders should return idential items. Validation and test dataloader
    # are at present not randomized, but checking those as well just in case.
    test_data_items_are_equal(["train_dataloader", "val_dataloader", "test_dataloader"])


def test_validate_columns(tmp_path: Path) -> None:
    _ = MockPandaSlidesGenerator(
        dest_data_path=tmp_path,
        mock_type=MockHistoDataType.FAKE,
        n_tiles=4,
        n_slides=10,
        n_channels=3,
        n_levels=3,
        tile_size=28,
        background_val=255,
        tiles_pos_type=TilesPositioningType.RANDOM,
    )
    usecols = [PandaDataset.SLIDE_ID_COLUMN, PandaDataset.MASK_COLUMN]
    with pytest.raises(ValueError, match=r"Expected columns"):
        _ = PandaDataset(root=tmp_path, dataframe_kwargs={"usecols": usecols})
    _ = PandaDataset(root=tmp_path, dataframe_kwargs={"usecols": usecols + [PandaDataset.METADATA_COLUMNS[1]]})
