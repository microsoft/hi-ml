#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import pandas as pd
from torchvision.datasets.vision import VisionDataset

from health_cpath.datasets.base_dataset import TilesDataset
from health_cpath.models.transforms import load_pil_image

from health_cpath.datasets.dataset_return_index import DatasetWithReturnIndex


class PandaTilesDataset(TilesDataset):
    """
    Dataset class for loading PANDA tiles.

    Iterating over this dataset returns a dictionary containing:
    - `'slide_id'` (str): parent slide ID (`'image_id'` in the PANDA dataset)
    - `'tile_id'` (str)
    - `'image'` (`PIL.Image`): RGB tile
    - `'mask'` (str): path to mask PNG file
    - `'tile_x'`, `'tile_y'` (int): top-right tile coordinates
    - `'data_provider'`, `'slide_isup_grade'`, `'slide_gleason_score'` (str): parent slide metadata
    """
    SPLIT_COLUMN = None  # PANDA does not have an official train/test split

    def __init__(self,
                 root: Path,
                 dataset_csv: Optional[Union[str, Path]] = None,
                 dataset_df: Optional[pd.DataFrame] = None,
                 occupancy_threshold: Optional[float] = None,
                 random_subset_fraction: Optional[float] = None,
                 label_column: str = "slide_isup_grade",
                 n_classes: int = 6) -> None:
        """
        :param root: Root directory of the dataset.
        :param dataset_csv: Full path to a dataset CSV file, containing at least
        `TILE_ID_COLUMN`, `SLIDE_ID_COLUMN`, and `IMAGE_COLUMN`. If omitted, the CSV will be read
        from `"{root}/{DEFAULT_CSV_FILENAME}"`.
        :param dataset_df: A potentially pre-processed dataframe in the same format as would be read
        from the dataset CSV file, e.g. after some filtering. If given, overrides `dataset_csv`.
        :occupancy_threshold: A value between 0-1 such that only tiles with occupancy > occupancy_threshold
        will be selected. If 0, all tiles are selected. If `None` (default), all tiles are selected.
        :random_subset_fraction: A value > 0 and <=1 such that this proportion of tiles will be randomly selected.
        If 1, all tiles are selected. If `None` (default), all tiles are selected.
        """
        super().__init__(root=Path(root),
                         dataset_csv=dataset_csv,
                         dataset_df=dataset_df,
                         train=None,
                         validate_columns=False,
                         label_column=label_column,
                         n_classes=n_classes)

        if occupancy_threshold is not None:
            if (occupancy_threshold < 0) or (occupancy_threshold > 1):
                raise ValueError(f"Occupancy threshold value {occupancy_threshold} should be in range 0-1.")
            dataset_df_filtered = self.dataset_df.loc[
                self.dataset_df['occupancy'] > occupancy_threshold
            ]
            self.dataset_df = dataset_df_filtered

        if random_subset_fraction is not None:
            if (random_subset_fraction <= 0) or (random_subset_fraction > 1):
                raise ValueError(f"Random subset fraction value {random_subset_fraction} should be > 0 and < = 1.")
            df_length_random_subset_fraction = round(len(self.dataset_df) * random_subset_fraction)
            dataset_df_filtered = self.dataset_df.sample(n=df_length_random_subset_fraction)
            self.dataset_df = dataset_df_filtered

        self.copy_coordinates_columns()
        self.validate_columns()


class PandaTilesDatasetReturnImageLabel(VisionDataset):
    """
    Any dataset used in SSL needs to return a tuple where the first element is the image and the second is a
    class label.
    """
    occupancy_threshold = 0
    random_subset_fraction = 1

    def __init__(self,
                 root: Path,
                 dataset_csv: Optional[Union[str, Path]] = None,
                 dataset_df: Optional[pd.DataFrame] = None,
                 transform: Optional[Callable] = None,
                 **kwargs: Any) -> None:
        super().__init__(root=root, transform=transform)

        self.base_dataset = PandaTilesDataset(root=root,
                                              dataset_csv=dataset_csv,
                                              dataset_df=dataset_df,
                                              occupancy_threshold=self.occupancy_threshold,
                                              random_subset_fraction=self.random_subset_fraction)

    def __getitem__(self, index: int) -> Tuple:  # type: ignore
        sample = self.base_dataset[index]
        # TODO change to a meaningful evaluation
        image = load_pil_image(sample[self.base_dataset.IMAGE_COLUMN])
        if self.transform:
            image = self.transform(image)
        # get binary label
        label = 0 if sample[self.base_dataset.label_column] == 0 else 1
        return image, label

    def __len__(self) -> int:
        return len(self.base_dataset)


class PandaTilesDatasetWithReturnIndex(DatasetWithReturnIndex, PandaTilesDatasetReturnImageLabel):
    """
    Any dataset used in SSL needs to inherit from DatasetWithReturnIndex as well as VisionData.
    This class is just a shorthand notation for this double inheritance. Please note that this class needs
    to override __getitem__(), this is why we need a separate PandaTilesDatasetReturnImageLabel.
    """
    @property
    def num_classes(self) -> int:
        return 2
