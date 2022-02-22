#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, Tuple

from health_ml.utils.split_dataset import DatasetSplits

from histopathology.datamodules.base_module import TilesDataModule
from histopathology.datasets.tcga_prad_tiles_dataset import TcgaPrad_TilesDataset


class TcgaPrad_TilesDataModule(TilesDataModule):
    """TcgaPrad_TilesDataModule is the child class of TilesDataModule specific to TCGA-PRAD dataset
    Method get_splits() returns the train, val, test splits from the TCGA-PRAD dataset
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def get_splits(self) -> Tuple[TcgaPrad_TilesDataset, TcgaPrad_TilesDataset, TcgaPrad_TilesDataset]:
        dataset = TcgaPrad_TilesDataset(self.root_path)
        splits = DatasetSplits.from_proportions(
            dataset.dataset_df.reset_index(),
            proportion_train=0.8,
            proportion_test=0.1,
            proportion_val=0.1,
            subject_column=dataset.TILE_ID_COLUMN,
            group_column=dataset.SLIDE_ID_COLUMN,
        )
        return (
            TcgaPrad_TilesDataset(self.root_path, dataset_df=splits.train),
            TcgaPrad_TilesDataset(self.root_path, dataset_df=splits.val),
            TcgaPrad_TilesDataset(self.root_path, dataset_df=splits.test),
        )
