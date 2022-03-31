#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np

from typing import Tuple, Optional
from histopathology.datamodules.base_module import SlidesDataModule
from histopathology.datasets.base_dataset import SlidesDataset
from health_azure.utils import PathOrString

METADATA_POSSIBLE_VALUES = {
    "data_provider": ["site_1", "site_2"],
    "isup_grade": [0, 4, 1, 3, 0, 5, 2, 5, 5, 4, 4],
    "gleason_score": ["0+0", "4+4", "3+3", "4+3", "negative", "4+5", "3+4", "5+4", "5+5", "5+3", "3+5"],
}
N_GLEASON_SCORES = 11


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

    def __init__(
        self, root: PathOrString, dataset_csv: Optional[PathOrString] = None, dataset_df: Optional[pd.DataFrame] = None
    ) -> None:
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


def create_mock_metadata_dataframe(tmp_path: str, n_samples: int = 4, seed: int = 42) -> None:
    """Create a mock dataframe with random metadata.

    :param tmp_path: A temporary directory to store the mock CSV file.
    :param n_samples: Number of random samples to generate, defaults to 4.
    :param seed: pseudorandom number generator seed to use for mocking random metadata, defaults to 42.
    """
    np.random.seed(seed)
    mock_metadata: dict = {col: [] for col in [MockSlidesDataset.IMAGE_COLUMN, *MockSlidesDataset.METADATA_COLUMNS]}
    for i in range(n_samples):
        rand_id = np.random.randint(0, N_GLEASON_SCORES)
        mock_metadata[MockSlidesDataset.IMAGE_COLUMN].append(f"_{i}")
        for key, val in METADATA_POSSIBLE_VALUES:
            mock_metadata[key].append(val[rand_id if len(val) == N_GLEASON_SCORES else np.random.randint(2)])
    df = pd.DataFrame(data=mock_metadata)
    df.to_csv(os.path.join(tmp_path, MockSlidesDataset.DEFAULT_CSV_FILENAME), index=False)
