#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Union, Optional
from health_cpath.datasets.base_dataset import SlidesDataset


class PandaColumns:
    SLIDE_ID = 'image_id'
    IMAGE = 'image'
    MASK = 'mask'
    METADATA = ('data_provider', 'isup_grade', 'gleason_score')


PANDA_CSV_FILENAME = "train.csv"


class PandaDataset(SlidesDataset):
    """Dataset class for loading files from the PANDA challenge dataset.

    Iterating over this dataset returns a dictionary following the `SlideKey` schema plus meta-data
    from the original dataset (`'data_provider'`, `'isup_grade'`, and `'gleason_score'`).

    Ref.: https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview
    """

    def __init__(
        self,
        root: Union[str, Path],
        dataset_csv: Optional[Union[str, Path]] = None,
        dataset_df: Optional[pd.DataFrame] = None,
        label_column: str = "isup_grade",
        n_classes: int = 6,
        dataframe_kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            root,
            dataset_csv,
            dataset_df,
            validate_columns=False,
            label_column=label_column,
            n_classes=n_classes,
            dataframe_kwargs=dataframe_kwargs,
            slide_id_column=PandaColumns.SLIDE_ID,
            image_column=PandaColumns.IMAGE,
            mask_column=PandaColumns.MASK,
            metadata_columns=PandaColumns.METADATA,
            default_csv_filename=PANDA_CSV_FILENAME,
        )
        # PANDA CSV does not come with paths for image and mask files
        slide_ids = self.dataset_df.index
        self.dataset_df[self.image_column] = "train_images/" + slide_ids + ".tiff"
        self.dataset_df[self.mask_column] = "train_label_masks/" + slide_ids + "_mask.tiff"
        self.validate_columns()
