#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd

from typing import Any, Tuple
import torch
from torchvision.utils import save_image

from health_ml.utils.split_dataset import DatasetSplits
from histopathology.datamodules.base_module import TilesDataModule
from histopathology.datasets.base_dataset import TilesDataset
from testhisto.mocks.base_datamodule import MockHistoDataGenerator


class MockTilesDataset(TilesDataset):
    """Mock and child class of SlidesDataset, to be used for testing purposes.
    It overrides the following, according to the PANDA cohort settings:

    :param LABEL_COLUMN: CSV column name for tile label set to "slide_isup_grade".
    :param SPLIT_COLUMN: CSV column name for train/test split (None for MockTiles data).
    :param N_CLASSES: Number of classes indexed in `LABEL_COLUMN`.
    """

    LABEL_COLUMN = "slide_isup_grade"
    SPLIT_COLUMN = None
    N_CLASSES = 6


class MockTilesDataModule(TilesDataModule):
    """Mock and child class of TilesDataModule, overrides get_splits so that it uses MockTilesDataset."""

    def get_splits(self) -> Tuple[MockTilesDataset, MockTilesDataset, MockTilesDataset]:
        dataset = MockTilesDataset(self.root_path)
        splits = DatasetSplits.from_proportions(
            dataset.dataset_df.reset_index(),
            proportion_train=0.8,
            proportion_test=0.1,
            proportion_val=0.1,
            subject_column=dataset.TILE_ID_COLUMN,
            group_column=dataset.SLIDE_ID_COLUMN,
        )
        return (
            MockTilesDataset(self.root_path, dataset_df=splits.train),
            MockTilesDataset(self.root_path, dataset_df=splits.val),
            MockTilesDataset(self.root_path, dataset_df=splits.test),
        )


class MockTilesGenerator(MockHistoDataGenerator):
    """Generator class to create mock tiles dataset on the fly. The tiles lay randomly on a wsi.

    :param TILE_ID_COLUMN: CSV column name for tile ID.
    :param SLIDE_ID_COLUMN: CSV column name for slide ID.
    :param IMAGE_COLUMN: CSV column name for relative path to image file.
    :param MASK_COLUMN: CSV column name for relative path to mask file.
    :param LABEL_COLUMN: CSV column name for tile label.
    :param TILE_X_COLUMN: CSV column name for horizontal tile coordinate.
    :param TILE_Y_COLUMN: CSV column name for vertical tile coordinate.
    :param DEFAULT_CSV_FILENAME: Default name of the dataset CSV at the dataset rood directory.
    :param DATA_PROVIDER: CSV column name for data provider .
    :param METADATA_POSSIBLE_VALUES: Possible values to be assigned to the dataset metadata. The values mapped to
    isup_grade are the possible gleason_scores.
    """

    SLIDE_ID_COLUMN = MockTilesDataset.SLIDE_ID_COLUMN
    TILE_ID_COLUMN = MockTilesDataset.TILE_ID_COLUMN
    IMAGE_COLUMN = MockTilesDataset.IMAGE_COLUMN
    MASK_COLUMN = "mask"
    TILE_X_COLUMN = MockTilesDataset.TILE_X_COLUMN
    TILE_Y_COLUMN = MockTilesDataset.TILE_Y_COLUMN
    OCCUPANCY = "occupancy"
    DEFAULT_CSV_FILENAME = MockTilesDataset.DEFAULT_CSV_FILENAME
    DATA_PROVIDER = "data_provider"
    ISUP_GRADE = "slide_isup_grade"
    GLEASON_SCORE = "gleason_score"

    N_CLASSES = MockTilesDataset.N_CLASSES

    METADATA_POSSIBLE_VALUES: dict = {
        DATA_PROVIDER: ["site_0", "site_1"],
        ISUP_GRADE: {
            0: ["0+0", "negative"],
            4: ["4+4", "5+3", "3+5"],
            1: ["3+3"],
            3: ["4+3"],
            2: ["3+4"],
            5: ["4+5", "5+4", "5+5"],
        },
    }

    METADATA_COLUMNS = tuple(METADATA_POSSIBLE_VALUES.keys())

    def __init__(self, img_size: int = 224, **kwargs: Any) -> None:
        self.img_size = img_size
        super().__init__(**kwargs)
        assert (
            self.n_slides >= self.N_CLASSES
        ), f"The number of slides should be >= self.N_CLASSES (i.e., {self.N_CLASSES})"
        assert (self.img_size // self.tile_size) ** 2 >= self.n_tiles, (
            f"The image of size {self.img_size} can't contain more than {(self.img_size // self.tile_size)**2} tiles."
            "Choose a number of tiles n_tiles <= (img_size // tile_size)**2 "
        )

    def create_mock_metadata_dataframe(self) -> pd.DataFrame:
        """Create a mock dataframe with random metadata."""
        df = pd.DataFrame(
            columns=[
                self.SLIDE_ID_COLUMN,
                self.TILE_ID_COLUMN,
                self.IMAGE_COLUMN,
                self.MASK_COLUMN,
                self.TILE_X_COLUMN,
                self.TILE_Y_COLUMN,
                self.OCCUPANCY,
                self.DATA_PROVIDER,
                self.ISUP_GRADE,
                self.GLEASON_SCORE,
            ]
        )
        n_tiles_side = self.img_size // self.tile_size
        total_n_tiles = n_tiles_side ** 2
        coords = [
            (k // n_tiles_side, k % n_tiles_side)
            for k in np.random.choice(total_n_tiles, size=self.n_tiles, replace=False)
        ]

        isup_grades = np.tile(
            list(self.METADATA_POSSIBLE_VALUES[self.ISUP_GRADE].keys()), self.n_slides // self.N_CLASSES + 1
        )

        for slide_id in range(self.n_slides):

            data_provider = np.random.choice(self.METADATA_POSSIBLE_VALUES[self.DATA_PROVIDER])
            isup_grade = isup_grades[slide_id]
            gleason_score = np.random.choice(self.METADATA_POSSIBLE_VALUES[self.ISUP_GRADE][isup_grade])
            # pick a random n_tiles for each slide
            n_tiles = np.random.randint(self.n_tiles // 2 + 1, 3 * self.n_tiles // 2)

            for tile_id in range(n_tiles):
                tile_x = coords[tile_id][0] * self.tile_size
                tile_y = coords[tile_id][1] * self.tile_size
                df.loc[(slide_id + 1) * tile_id] = [
                    f"_{slide_id}",
                    f"_{slide_id}.{tile_x}x_{tile_y}y",
                    f"_{slide_id}/train_images/{tile_x}x_{tile_y}y.png",
                    f"_{slide_id}/train_images/{tile_x}x_{tile_y}y_mask.png",
                    tile_x,
                    tile_y,
                    1.0,
                    data_provider,
                    isup_grade,
                    gleason_score,
                ]
        df.to_csv(os.path.join(self.tmp_path, self.DEFAULT_CSV_FILENAME), index=False)
        return df

    def generate_mock_histo_data(self) -> None:
        for _, row in self.dataframe.iterrows():
            slide_dir = self.tmp_path / f"{row[self.SLIDE_ID_COLUMN]}/train_images"
            os.makedirs(slide_dir, exist_ok=True)
            tiles, _ = next(iter(self.dataloader))
            for tile in tiles:
                save_image(tile * 255, str(self.tmp_path / row[self.IMAGE_COLUMN]))
                random_mask = torch.randint(0, 256, size=(self.n_channels, self.tile_size, self.tile_size))
                save_image(random_mask.float(), str(self.tmp_path / row[self.MASK_COLUMN]))
