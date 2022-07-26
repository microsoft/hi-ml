#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from health_cpath.datasets.base_dataset import TEST_SPLIT_LABEL, TRAIN_SPLIT_LABEL

from health_cpath.utils.naming import TileKey


class TilesDatasetColumns:
    """
    Columns in the dataset CSV.
    """
    def __init__(self,
                 slide_id: str = "slide_id",
                 tile_id: str = "tile_id",
                 image: str = "image",
                 path: str = "image_path",
                 case_id: str = "case_id",
                 label: str = "label",
                 split: Optional[str] = "split",
                 tile_x: Optional[str] = "tile_x",
                 tile_y: Optional[str] = "tile_y",
                 ) -> None:
        """_summary_

        :param slide_id: CSV column name for tile ID., defaults to "slide_id"
        :param tile_id: CSV column name for slide ID, defaults to "tile_id"
        :param image: CSV column name for relative path to image file, defaults to "image"
        :param path: CSV column name for relative path to image file. Replicated to propagate the path to the batch,
        defaults to "image_path"
        :param label: CSV column name for tile label, defaults to "label"
        :param split: CSV column name for train/test split (optional), defaults to "split"
        :param tile_x: CSV column name for horizontal tile coordinate (optional), defaults to "tile_x"
        :param tile_y: CSV column name for vertical tile coordinate (optional), defaults to "tile_y"
        """
        self.slide_id = slide_id
        self.tile_id = tile_id
        self.image = image
        self.path = path
        self.case_id = case_id
        self.label = label
        self.split = split
        self.tile_x = tile_x
        self.tile_y = tile_y

    def get_validation_columns(self) -> List[Union[str, Optional[str]]]:
        return [self.slide_id, self.image, self.label, self.split, self.tile_x, self.tile_y]


class TilesDataset(Dataset):
    """Base class for datasets of WSI tiles, iterating dictionaries of image paths and metadata."""

    DEFAULT_CSV_FILENAME: str = "dataset.csv"

    def __init__(self,
                 name: str,
                 root: Union[str, Path],
                 tiles_columns: TilesDatasetColumns,
                 n_classes: int = 1,
                 dataset_csv: Optional[Union[str, Path]] = None,
                 dataset_df: Optional[pd.DataFrame] = None,
                 train: Optional[bool] = None,
                 occupancy_threshold: Optional[float] = None,
                 random_subset_fraction: Optional[float] = None) -> None:
        """
        :param name: Name of the dataset (e.g. PANDA, TCGA_Crck, ...).
        :param root: Root directory of the dataset.
        :param tiles_columns: TilesDatasetColumns object containing the names of the columns in the dataset CSV.
        :param n_classes: Number of classes indexed in `n_classes`. Default is 1 for binary classification.
        :param dataset_csv: Full path to a dataset CSV file, containing at least
        `tile_id`, `slide_id`, and `image` columns. If omitted, the CSV will be read
        from `"{root}/{DEFAULT_CSV_FILENAME}"`.
        :param dataset_df: A potentially pre-processed dataframe in the same format as would be read
        from the dataset CSV file, e.g. after some filtering. If given, overrides `dataset_csv`.
        :param train: If `True`, loads only the training split (resp. `False` for test split). By
        default (`None`), loads the entire dataset as-is.
        """
        self.name = name
        self.root_dir = Path(root)
        self.columns = tiles_columns
        self.n_classes = n_classes

        if self.columns.split is None and train is not None:
            raise ValueError("Train/test split was specified but dataset has no split column")

        if dataset_df is not None:
            self.dataset_csv = None
        else:
            self.dataset_csv = dataset_csv or self.root_dir / self.DEFAULT_CSV_FILENAME
            dataset_df = pd.read_csv(self.dataset_csv)

        dataset_df = dataset_df.set_index(self.columns.tile_id)
        if train is None:
            self.dataset_df = dataset_df
        else:
            split = TRAIN_SPLIT_LABEL if train else TEST_SPLIT_LABEL
            self.dataset_df = dataset_df[dataset_df[self.columns.split] == split]

        self.filter_dataset_by_occupancy(occupancy_threshold)
        self.randomly_fraction_dataset(random_subset_fraction)
        self.replicate_coordinates()
        self.validate_columns()

    def filter_dataset_by_occupancy(self, occupancy_threshold: Optional[float] = None) -> None:
        if occupancy_threshold is not None:
            if (occupancy_threshold < 0) or (occupancy_threshold > 1):
                raise ValueError(f"Occupancy threshold value {occupancy_threshold} should be in range 0-1.")
            dataset_df_filtered = self.dataset_df.loc[
                self.dataset_df["occupancy"] > occupancy_threshold
            ]
            self.dataset_df = dataset_df_filtered

    def randomly_fraction_dataset(self, random_subset_fraction: Optional[float] = None) -> None:
        if random_subset_fraction is not None:
            if (random_subset_fraction <= 0) or (random_subset_fraction > 1):
                raise ValueError(f"Random subset fraction value {random_subset_fraction} should be > 0 and < = 1.")
            df_length_random_subset_fraction = round(len(self.dataset_df) * random_subset_fraction)
            dataset_df_filtered = self.dataset_df.sample(n=df_length_random_subset_fraction)
            self.dataset_df = dataset_df_filtered

    def replicate_coordinates(self) -> None:
        """ Copy columns "left" --> "tile_x" and "top" --> "tile_y" to be consistent with TilesDataset `tile_x`
        and `tile_y` columns.
        """
        if TileKey.TILE_LEFT in self.dataset_df.columns and self.columns.tile_x:
            self.dataset_df[self.columns.tile_x] = self.dataset_df[TileKey.TILE_LEFT]
        if TileKey.TILE_TOP in self.dataset_df.columns and self.columns.tile_y:
            self.dataset_df[self.columns.tile_y] = self.dataset_df[TileKey.TILE_TOP]

    def validate_columns(self) -> None:
        """Check that loaded dataframe contains expected columns, raises `ValueError` otherwise.

        If the constructor is overloaded in a subclass, you can pass `validate_columns=False` and
        call `validate_columns()` after creating derived columns, for example.
        """
        columns = self.columns.get_validation_columns()
        columns_not_found = []
        for column in columns:
            if column is not None and column not in self.dataset_df.columns:
                columns_not_found.append(column)
        if len(columns_not_found) > 0:
            raise ValueError(f"Expected columns '{columns_not_found}' not found in the dataframe")

    def __len__(self) -> int:
        return self.dataset_df.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        tile_id = self.dataset_df.index[index]
        sample = {
            self.columns.tile_id: tile_id,
            **self.dataset_df.loc[tile_id].to_dict()
        }
        sample[self.columns.image] = str(self.root_dir / sample.pop(self.columns.image))
        # we're replicating this column because we want to propagate the path to the batch
        sample[self.columns.path] = sample[self.columns.image]
        return sample

    @property
    def slide_ids(self) -> pd.Series:
        return self.dataset_df[self.columns.slide_id]

    def get_slide_labels(self) -> pd.Series:
        return self.dataset_df.groupby(self.columns.slide_id)[self.columns.label].agg(pd.Series.mode)

    def get_class_weights(self) -> torch.Tensor:
        slide_labels = self.get_slide_labels()
        classes = np.unique(slide_labels)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=slide_labels)
        return torch.as_tensor(class_weights)
