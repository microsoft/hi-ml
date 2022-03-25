#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, Tuple

from health_ml.utils.split_dataset import DatasetSplits

from histopathology.datamodules.base_module import TilesDataModule
from histopathology.datasets.panda_tiles_dataset import PandaTilesDataset


class PandaTilesDataModule(TilesDataModule):
    """ PandaTilesDataModule is the child class of TilesDataModule specific to PANDA dataset
    Method get_splits() returns the train, val, test splits from the PANDA dataset
    """

    def get_splits(self) -> Tuple[PandaTilesDataset, PandaTilesDataset, PandaTilesDataset]:
        dataset = PandaTilesDataset(self.root_path)
        splits = DatasetSplits.from_proportions(dataset.dataset_df.reset_index(),
                                                proportion_train=.8,
                                                proportion_test=.1,
                                                proportion_val=.1,
                                                subject_column=dataset.TILE_ID_COLUMN,
                                                group_column=dataset.SLIDE_ID_COLUMN)

        if self.crossval_count > 1:
            # Function get_k_fold_cross_validation_splits() will concatenate train and val splits
            splits = splits.get_k_fold_cross_validation_splits(self.crossval_count)[self.crossval_index]

        return (PandaTilesDataset(self.root_path, dataset_df=splits.train),
                PandaTilesDataset(self.root_path, dataset_df=splits.val),
                PandaTilesDataset(self.root_path, dataset_df=splits.test))
