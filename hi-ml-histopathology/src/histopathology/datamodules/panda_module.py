#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Tuple, Union

from health_ml.utils.split_dataset import DatasetSplits

from histopathology.datamodules.base_module import SlidesDataModule, TilesDataModule
from histopathology.datasets.panda_dataset import PandaDataset
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


class SubPandaTilesDataModule(TilesDataModule):
    """ subPandaTilesDataModule is a child class of TilesDataModule specific to PANDA dataset. The difference with `PandaTilesDataModule` is that
    Method get_splits() returns the train, val, test splits from a subset of the PANDA dataset specified by
    train/validation dataframes.
    """

    def __init__(self, train_csv: Union[str, Path], val_csv: Union[str, Path], **kwargs: Any) -> None:
        self.train_csv = train_csv
        self.val_csv = val_csv
        super().__init__(**kwargs)

    def get_splits(self) -> Tuple[PandaTilesDataset, PandaTilesDataset, PandaTilesDataset]:
        return (
            PandaTilesDataset(self.root_path, self.train_csv),
            PandaTilesDataset(self.root_path, self.val_csv),
            PandaTilesDataset(self.root_path, self.val_csv),
        )


class PandaSlidesDataModule(SlidesDataModule):
    """ PandaSlidesDataModule is the child class of SlidesDataModule specific to PANDA dataset
    Method get_splits() returns the train, val, test splits from the PANDA dataset
    """

    def get_splits(self) -> Tuple[PandaDataset, PandaDataset, PandaDataset]:
        dataset = PandaDataset(self.root_path)
        splits = DatasetSplits.from_proportions(dataset.dataset_df.reset_index(),
                                                proportion_train=.8,
                                                proportion_test=.1,
                                                proportion_val=.1,
                                                subject_column=dataset.TILE_ID_COLUMN,
                                                group_column=dataset.SLIDE_ID_COLUMN)

        if self.crossval_count > 1:
            # Function get_k_fold_cross_validation_splits() will concatenate train and val splits
            splits = splits.get_k_fold_cross_validation_splits(self.crossval_count)[self.crossval_index]

        return (PandaDataset(self.root_path, dataset_df=splits.train),
                PandaDataset(self.root_path, dataset_df=splits.val),
                PandaDataset(self.root_path, dataset_df=splits.test))


class SubPandaSlidesDataModule(PandaSlidesDataModule):
    """ PandaSlidesDataModule is the child class of SlidesDataModule specific to PANDA dataset
    Method get_splits() returns the train, val, test splits from the PANDA dataset
    """

    def __init__(self, train_csv: Union[str, Path], val_csv: Union[str, Path], **kwargs: Any) -> None:
        self.train_csv = train_csv
        self.val_csv = val_csv
        super().__init__(**kwargs)

    def get_splits(self) -> Tuple[PandaDataset, PandaDataset, PandaDataset]:
        return (
            PandaDataset(self.root_path, self.train_csv),
            PandaDataset(self.root_path, self.val_csv),
            PandaDataset(self.root_path, self.val_csv),
        )
