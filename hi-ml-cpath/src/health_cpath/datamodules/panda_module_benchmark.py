#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Configuration of data modules for PANDA DeepMIL experiments.
Ref: Myronenko et al. 2021 (https://link.springer.com/chapter/10.1007/978-3-030-87237-3_32)
"""

from typing import Tuple

from health_ml.utils.split_dataset import DatasetSplits

from health_cpath.datamodules.base_module import SlidesDataModule
from health_cpath.datasets.panda_dataset import PandaDataset


class PandaSlidesDataModuleBenchmark(SlidesDataModule):
    """ PandaSlidesDataModuleBenchmark is the child class of SlidesDataModule specific to PANDA dataset
    Method get_splits() returns the train, val, test splits from the PANDA dataset
    """

    def get_splits(self) -> Tuple[PandaDataset, PandaDataset, PandaDataset]:
        dataset = PandaDataset(self.root_path)
        splits = DatasetSplits.from_proportions(dataset.dataset_df.reset_index(),
                                                proportion_train=0.8,
                                                proportion_test=0.0,
                                                proportion_val=0.2,
                                                subject_column=dataset.SLIDE_ID_COLUMN)

        if self.crossval_count > 1:
            # Function get_k_fold_cross_validation_splits() will concatenate train and val splits
            splits = splits.get_k_fold_cross_validation_splits(self.crossval_count)[self.crossval_index]

        return (PandaDataset(self.root_path, dataset_df=splits.train),
                PandaDataset(self.root_path, dataset_df=splits.val),
                PandaDataset(self.root_path, dataset_df=splits.val))
