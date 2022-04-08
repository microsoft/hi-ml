#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Tuple
import numpy as np
import pandas as pd
from histopathology.datamodules.base_module import TilesDataModule
from histopathology.datasets.base_dataset import TilesDataset
from testhisto.utils.utils_base_datamodule import MockHistoDataGenerator


class MockTilesDataset(TilesDataset):
    LABEL_COLUMN = "slide_isup_grade"
    SPLIT_COLUMN = None
    N_CLASSES = 6


class MockTilesDataModule(TilesDataModule):
    def get_splits(self) -> Tuple[MockTilesDataset, MockTilesDataset, MockTilesDataset]:
        df = MockTilesDataset(self.root_path).dataset_df
        df = df.reset_index()
        split_dfs = (
            df[df[MockTilesDataset.SPLIT_COLUMN] == MockTilesDataset.TRAIN_SPLIT_LABEL],
            df[df[MockTilesDataset.SPLIT_COLUMN] == MockTilesDataset.VAL_SPLIT_LABEL],
            df[df[MockTilesDataset.SPLIT_COLUMN] == MockTilesDataset.TEST_SPLIT_LABEL],
        )
        return tuple(
            MockTilesDataset(self.root_path, dataset_df=split_df)  # type: ignore
            for split_df in split_dfs
        )


class MockTilesGenerator(MockHistoDataGenerator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def create_mock_metadata_dataframe(self) -> None:
        slide_ids = np.random.randint(self.n_slides, size=self.n_tiles)
        slide_labels = np.random.randint(self.N_CLASSES, size=self.n_slides)
        tile_labels = slide_labels[slide_ids]
        split_labels = [
            MockTilesDataset.TRAIN_SPLIT_LABEL,
            MockTilesDataset.VAL_SPLIT_LABEL,
            MockTilesDataset.TEST_SPLIT_LABEL,
        ]
        slide_splits = np.random.choice(split_labels, size=self.n_slides)
        tile_splits = slide_splits[slide_ids]

        df = pd.DataFrame()
        df[MockTilesDataset.TILE_ID_COLUMN] = np.arange(self.n_tiles)
        df[MockTilesDataset.SLIDE_ID_COLUMN] = slide_ids
        df[MockTilesDataset.LABEL_COLUMN] = tile_labels
        df[MockTilesDataset.SPLIT_COLUMN] = tile_splits
        df[MockTilesDataset.IMAGE_COLUMN] = [f"{tile_splits[i]}/{i:06d}.png" for i in range(self.n_tiles)]

        return df
