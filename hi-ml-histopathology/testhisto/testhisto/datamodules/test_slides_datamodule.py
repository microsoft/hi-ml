#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pandas as pd

from pathlib import Path
from typing import Any, Tuple, Union, Optional

import pytest
from torch.utils.data import dataloader
from histopathology.datamodules.base_module import SlidesDataModule
from histopathology.datasets.base_dataset import SlidesDataset

MOCK_DATA_PATH = "../../test_data/whole_slide_images/pathmnist"


class MockSlidesDataset(SlidesDataset):
    DEFAULT_CSV_FILENAME = "dataset.csv"

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
    def __init__(**kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _get_slides_dataset_class(self) -> None:
        return MockSlidesDataset

    def get_splits(self) -> Tuple[MockSlidesDataset, MockSlidesDataset, MockSlidesDataset]:
        return (MockSlidesDataset(self.root_path), MockSlidesDataset(self.root_path), MockSlidesDataset(self.root_path))


@pytest.mark.parametrize("level", [(0,), (1,), (2,)])
def test_tiling_on_the_fly(level: int = 0) -> None:
    datamodule = MockSlidesDataModule(root_path=MOCK_DATA_PATH, batch_size=1, tile_count=16, tile_size=28, level=level)
    dataloader = datamodule._get_dataloader(stage="train", shuffle=False)
    sample = next(iter(dataloader))
    tiles = sample["image"]
    wsi_id = sample["slide_id"].split("/")[0]
    patches = np.array([])
    assert tiles.shape == (1, 16, 3, 28, 28)


def test_overlapping_tiling()->None:
    pass
