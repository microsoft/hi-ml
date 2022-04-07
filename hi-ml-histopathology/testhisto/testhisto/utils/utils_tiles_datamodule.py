#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Tuple
import numpy as np
import pandas as pd
from histopathology.datamodules.base_module import TilesDataModule
from histopathology.datasets.base_dataset import TilesDataset


class MockTilesDataset(TilesDataset):
    TILE_X_COLUMN = TILE_Y_COLUMN = None
    TRAIN_SPLIT_LABEL = 'train'
    VAL_SPLIT_LABEL = 'val'
    TEST_SPLIT_LABEL = 'test'


class MockTilesDataModule(TilesDataModule):
    def get_splits(self) -> Tuple[MockTilesDataset, MockTilesDataset, MockTilesDataset]:
        df = MockTilesDataset(self.root_path).dataset_df
        df = df.reset_index()
        split_dfs = (df[df[MockTilesDataset.SPLIT_COLUMN] == MockTilesDataset.TRAIN_SPLIT_LABEL],
                     df[df[MockTilesDataset.SPLIT_COLUMN] == MockTilesDataset.VAL_SPLIT_LABEL],
                     df[df[MockTilesDataset.SPLIT_COLUMN] == MockTilesDataset.TEST_SPLIT_LABEL])
        return tuple(MockTilesDataset(self.root_path, dataset_df=split_df)  # type: ignore
                     for split_df in split_dfs)
        
        