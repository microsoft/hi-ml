#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from health_cpath.datasets.base_dataset import DEFAULT_LABEL_COLUMN, SlidesDataset


TCGA_PRAD_DATASET_FILE = "dataset.csv"


class TcgaColumns:
    """Column names for TCGA dataset CSV files."""

    IMAGE = "image_path"
    LABEL1 = "label1"
    LABEL2 = "label2"


class TcgaPradDataset(SlidesDataset):
    """Dataset class for loading TCGA-PRAD slides.

    Iterating over this dataset returns a dictionary containing:
    - `'slide_id'` (str)
    - `'case_id'` (str)
    - `'image_path'` (str): absolute slide image path
    - `'label'` (int, 0 or 1): label for predicting positive or negative
    """

    def __init__(
        self,
        root: Union[str, Path],
        dataset_csv: Optional[Union[str, Path]] = None,
        dataset_df: Optional[pd.DataFrame] = None,
        label_column: str = DEFAULT_LABEL_COLUMN,
    ) -> None:
        """
        :param root: Root directory of the dataset.
        :param dataset_csv: Full path to a dataset CSV file. If omitted, the CSV will be read from
        `"{root}/{DEFAULT_CSV_FILENAME}"`.
        :param dataset_df: A potentially pre-processed dataframe in the same format as would be read
        from the dataset CSV file, e.g. after some filtering. If given, overrides `dataset_csv`.
        """
        super().__init__(
            root,
            dataset_csv,
            dataset_df,
            validate_columns=False,
            label_column=label_column,
            default_csv_filename=TCGA_PRAD_DATASET_FILE,
            image_column=TcgaColumns.IMAGE,
        )
        # Example of how to define a custom label column from existing columns:
        self.dataset_df[self.label_column] = (
            self.dataset_df[TcgaColumns.LABEL1] | self.dataset_df[TcgaColumns.LABEL2]
        ).astype(
            int
        )  # noqa: W503
        self.validate_columns()
