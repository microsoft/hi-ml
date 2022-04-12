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
    :param MASK_COLUMN: CSV column name for relative path to mask file.
    :param DATA_PROVIDER: CSV column name for data provider.
    :param ISUP_GRADE: CSV column name for isup grade.
    :param GLEASON_SCORE: CSV column name for gleason score.
    """

    LABEL_COLUMN = "isup_grade"
    SPLIT_COLUMN = None
    N_CLASSES = 6
    MASK_COLUMN = "mask"
    OCCUPANCY = "occupancy"
    DATA_PROVIDER = "data_provider"
    ISUP_GRADE = "isup_grade"
    GLEASON_SCORE = "gleason_score"


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
    """Generator class to create mock tiles dataset on the fly. The tiles lay randomly on a wsi."""

    def __init__(self, img_size: int = 224, **kwargs: Any) -> None:
        """
        :param img_size: The whole slide image resolution, defaults to 224.
        :param kwargs: Same params passed to MockHistoDataGenerator.
        """
        self.img_size = img_size
        super().__init__(**kwargs)
        assert (
            self.n_slides >= MockTilesDataset.N_CLASSES
        ), f"The number of slides should be >= MockTilesDataset.N_CLASSES (i.e., {MockTilesDataset.N_CLASSES})"
        assert (self.img_size // self.tile_size) ** 2 >= self.n_tiles, (
            f"The image of size {self.img_size} can't contain more than {(self.img_size // self.tile_size)**2} tiles."
            f"Choose a number of tiles 0 < n_tiles <= {(self.img_size // self.tile_size)**2} "
        )

    def create_mock_metadata_dataframe(self) -> pd.DataFrame:
        """Create a mock dataframe with random metadata."""
        csv_columns = [
            MockTilesDataset.SLIDE_ID_COLUMN,
            MockTilesDataset.TILE_ID_COLUMN,
            MockTilesDataset.IMAGE_COLUMN,
            MockTilesDataset.MASK_COLUMN,
            MockTilesDataset.TILE_X_COLUMN,
            MockTilesDataset.TILE_Y_COLUMN,
            MockTilesDataset.OCCUPANCY,
            MockTilesDataset.DATA_PROVIDER,
            MockTilesDataset.ISUP_GRADE,
            MockTilesDataset.GLEASON_SCORE,
        ]
        mock_metadata: dict = {col: [] for col in csv_columns}

        n_tiles_side = self.img_size // self.tile_size
        total_n_tiles = n_tiles_side ** 2

        # This is to make sure that the dataset contains at least one sample from each isup grade class.
        isup_grades = np.tile(
            list(self.METADATA_POSSIBLE_VALUES[MockTilesDataset.ISUP_GRADE].keys()),
            self.n_slides // MockTilesDataset.N_CLASSES + 1,
        )

        for slide_id in range(self.n_slides):

            data_provider = np.random.choice(self.METADATA_POSSIBLE_VALUES[MockTilesDataset.DATA_PROVIDER])
            isup_grade = isup_grades[slide_id]
            gleason_score = np.random.choice(self.METADATA_POSSIBLE_VALUES[MockTilesDataset.ISUP_GRADE][isup_grade])

            # pick a random n_tiles for each slide
            n_tiles = np.random.randint(self.n_tiles // 2 + 1, 3 * self.n_tiles // 2)

            coords = [
                (k // n_tiles_side, k % n_tiles_side)
                for k in np.random.choice(total_n_tiles, size=n_tiles, replace=False)
            ]

            for tile_id in range(n_tiles):
                tile_x = coords[tile_id][0] * self.tile_size
                tile_y = coords[tile_id][1] * self.tile_size
                mock_metadata[MockTilesDataset.SLIDE_ID_COLUMN].append(f"_{slide_id}")
                mock_metadata[MockTilesDataset.TILE_ID_COLUMN].append(f"_{slide_id}.{tile_x}x_{tile_y}y")
                mock_metadata[MockTilesDataset.IMAGE_COLUMN].append(f"_{slide_id}/train_images/{tile_x}x_{tile_y}y.png")
                mock_metadata[MockTilesDataset.MASK_COLUMN].append(
                    f"_{slide_id}/train_images/{tile_x}x_{tile_y}y_mask.png"
                )
                mock_metadata[MockTilesDataset.TILE_X_COLUMN].append(tile_x)
                mock_metadata[MockTilesDataset.TILE_Y_COLUMN].append(tile_y)
                mock_metadata[MockTilesDataset.OCCUPANCY].append(1.0)
                mock_metadata[MockTilesDataset.DATA_PROVIDER].append(data_provider)
                mock_metadata[MockTilesDataset.ISUP_GRADE].append(isup_grade)
                mock_metadata[MockTilesDataset.GLEASON_SCORE].append(gleason_score)

        df = pd.DataFrame(data=mock_metadata)
        df.to_csv(os.path.join(self.tmp_path, MockTilesDataset.DEFAULT_CSV_FILENAME), index=False)
        return df

    def generate_mock_histo_data(self) -> None:
        iterator = iter(self.dataloader) if self.dataloader else None
        for _, row in self.dataframe.iterrows():
            slide_dir = self.tmp_path / f"{row[MockTilesDataset.SLIDE_ID_COLUMN]}/train_images"
            os.makedirs(slide_dir, exist_ok=True)
            tiles, _ = next(iterator) if iterator else (None, None)
            for tile in tiles:
                save_image(tile * 255, str(self.tmp_path / row[MockTilesDataset.IMAGE_COLUMN]))
                random_mask = torch.randint(0, 256, size=(self.n_channels, self.tile_size, self.tile_size))
                save_image(random_mask.float(), str(self.tmp_path / row[MockTilesDataset.MASK_COLUMN]))
