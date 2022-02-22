#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import pandas as pd
from torchvision.datasets.vision import VisionDataset

from histopathology.datasets.base_dataset import TilesDataset
from histopathology.models.transforms import load_pil_image

from SSL.data.dataset_cls_utils import DataClassBaseWithReturnIndex


class TcgaPrad_TilesDataset(TilesDataset):
    """
    Dataset class for loading TCGA tiles.

    Iterating over this dataset returns a dictionary containing:
    - `'slide_id'` (str): parent slide ID
    - `'tile_id'` (str): tile ID
    - `'image'` (`PIL.Image`): RGB tile
    - `'label'` (str): negative (0) vs positive (1)
    - `'tile_x'`, `'tile_y'` (int): top-right tile coordinates
    - `'occupancy'`, `'slide_case_id'`, `'slide_project'`,
       `'slide_primary_site'`, `'slide_gender'`, `'slide_age_at_diagnosis'`,
       `slide_days_to_death'`, `'slide_vital_status'`, `'slide_primary_diagnosis'`,
       `'slide_ethnicity'`, `'slide_race'`, `'slide_brca1_mutation'`, `'slide_brca2_mutation'`,
       `'slide_brca1_tag'`, `'slide_brca2_tag'`(str): parent slide metadata
    """

    SPLIT_COLUMN = None  # TCGA-PRAD does not have an official train/test split
    # This dataset conforms to all other defaults in TilesDataset

    _RELATIVE_ROOT_FOLDER = Path("TCGA-PRAD_tiles_20211209-194609")

    def __init__(
        self,
        root: Path,
        dataset_csv: Optional[Union[str, Path]] = None,
        dataset_df: Optional[pd.DataFrame] = None,
        occupancy_threshold: Optional[float] = None,
    ) -> None:
        super().__init__(
            root=Path(root) / self._RELATIVE_ROOT_FOLDER, dataset_csv=dataset_csv, dataset_df=dataset_df, train=None
        )
        if occupancy_threshold is not None:
            self.dataset_df: pd.DataFrame
            dataset_df_filtered = self.dataset_df.loc[self.dataset_df["occupancy"] > occupancy_threshold]
            self.dataset_df = dataset_df_filtered


class TcgaPrad_TilesDatasetReturnImageLabel(VisionDataset):
    """
    Any dataset used in SSL needs to return a tuple where the first element is the image and the second is a
    class label.
    """

    # declaring `occupancy_threshold`` as a class attribute, can be overridden from SSL container
    occupancy_threshold = 0

    def __init__(
        self,
        root: Path,
        dataset_csv: Optional[Union[str, Path]] = None,
        dataset_df: Optional[pd.DataFrame] = None,
        transform: Optional[Callable] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(root=root, transform=transform)

        self.base_dataset = TcgaPrad_TilesDataset(
            root=root, dataset_csv=dataset_csv, dataset_df=dataset_df, occupancy_threshold=self.occupancy_threshold
        )

    def __getitem__(self, index: int) -> Tuple:  # type: ignore
        sample = self.base_dataset[index]
        # TODO change to a meaningful evaluation
        image = load_pil_image(sample[self.base_dataset.IMAGE_COLUMN])
        if self.transform:
            image = self.transform(image)
        return image, sample[self.base_dataset.LABEL_COLUMN]

    def __len__(self) -> int:
        return len(self.base_dataset)


class TcgaPrad_TilesDatasetWithReturnIndex(DataClassBaseWithReturnIndex, TcgaPrad_TilesDatasetReturnImageLabel):
    """
    Any dataset used in SSL needs to inherit from DataClassBaseWithReturnIndex as well as VisionData.
    This class is just a shorthand notation for this double inheritance. Please note that this class needs
    to override __getitem__(), this is why we need a separate TcgaPrad_TilesDatasetReturnImageLabel.
    """

    @property
    def num_classes(self) -> int:
        return 2
