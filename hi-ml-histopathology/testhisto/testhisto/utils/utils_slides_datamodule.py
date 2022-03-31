#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pandas as pd

from typing import Tuple, Optional
from histopathology.datamodules.base_module import SlidesDataModule
from histopathology.datasets.base_dataset import SlidesDataset
from health_azure.utils import PathOrString


class MockSlidesDataset(SlidesDataset):
    """Mock and child class of SlidesDataset, to be used for testing purposes.
    It overrides the following, according to the PANDA cohort setting:

    :param SLIDE_ID_COLUMN: CSV column name for slide ID set to "image_id".
    :param LABEL_COLUMN: CSV column name for tile label set to "isup_grade".
    :param METADATA_COLUMNS: Column names for all the metadata available on the CSV dataset file.
    """
    SLIDE_ID_COLUMN = "image_id"
    LABEL_COLUMN = "isup_grade"

    METADATA_COLUMNS = ("data_provider", "isup_grade", "gleason_score")

    def __init__(self,
                 root: PathOrString,
                 dataset_csv: Optional[PathOrString] = None,
                 dataset_df: Optional[pd.DataFrame] = None) -> None:
        """
        :param root: Root directory of the dataset.
        :param dataset_csv: Full path to a dataset CSV file, containing at least
        `TILE_ID_COLUMN`, `SLIDE_ID_COLUMN`, and `IMAGE_COLUMN`. If omitted, the CSV will be read
        from `"{root}/{DEFAULT_CSV_FILENAME}"`.
        :param dataset_df: A potentially pre-processed dataframe in the same format as would be read
        from the dataset CSV file, e.g. after some filtering. If given, overrides `dataset_csv`.
        """
        super().__init__(root, dataset_csv, dataset_df, validate_columns=False)
        slide_ids = self.dataset_df.index
        self.dataset_df[self.IMAGE_COLUMN] = slide_ids + ".tiff"
        self.validate_columns()


class MockSlidesDataModule(SlidesDataModule):
    """Mock and child class of SlidesDataModule. It overrides get_splits so that it uses MockSlidesDataset.
    """
    def get_splits(self) -> Tuple[MockSlidesDataset, MockSlidesDataset, MockSlidesDataset]:
        return (MockSlidesDataset(self.root_path), MockSlidesDataset(self.root_path), MockSlidesDataset(self.root_path))