#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Tuple

from health_ml.utils.split_dataset import DatasetSplits

from health_cpath.datamodules.base_module import TilesDataModule
from health_cpath.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDataset


class TcgaCrckTilesDataModule(TilesDataModule):
    """ TcgaCrckTilesDataModule is the child class of TilesDataModule specific to TCGA-Crck dataset
    """

    def get_splits(self) -> Tuple[TcgaCrck_TilesDataset, TcgaCrck_TilesDataset, TcgaCrck_TilesDataset]:
        """
        :return: the train, val, test splits from the TCGA-Crck dataset
        """
        trainval_dataset = TcgaCrck_TilesDataset(self.root_path, train=True)
        splits = DatasetSplits.from_proportions(trainval_dataset.dataset_df.reset_index(),
                                                proportion_train=0.8,
                                                proportion_test=0.0,
                                                proportion_val=0.2,
                                                subject_column=trainval_dataset.TILE_ID_COLUMN,
                                                group_column=trainval_dataset.SLIDE_ID_COLUMN,
                                                random_seed=5)

        if self.crossval_count > 1:
            # Function get_k_fold_cross_validation_splits() will concatenate train and val splits
            splits = splits.get_k_fold_cross_validation_splits(self.crossval_count)[self.crossval_index]

        return (TcgaCrck_TilesDataset(self.root_path, dataset_df=splits.train),
                TcgaCrck_TilesDataset(self.root_path, dataset_df=splits.val),
                TcgaCrck_TilesDataset(self.root_path, train=False))
