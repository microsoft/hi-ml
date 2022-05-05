#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import pandas as pd

from typing import Any
import torch
from torchvision.utils import save_image

from histopathology.datasets.panda_tiles_dataset import PandaTilesDataset
from testhisto.mocks.base_data_generator import MockHistoDataGenerator, MockHistoDataType


class MockPandaTilesGenerator(MockHistoDataGenerator):
    """Generator class to create mock tiles dataset on the fly. The tiles are positioned randomly in a wsi grid."""

    def __init__(self, img_size: int = 224, **kwargs: Any) -> None:
        """
        :param img_size: The whole slide image resolution, defaults to 224.
        :param kwargs: Same params passed to MockHistoDataGenerator.
        """
        self.img_size = img_size
        super().__init__(**kwargs)

    def validate(self) -> None:
        assert (
            self.n_slides >= PandaTilesDataset.N_CLASSES
        ), f"The number of slides should be >= N_CLASSES (i.e., {PandaTilesDataset.N_CLASSES})"
        assert (self.img_size // self.tile_size) ** 2 >= self.n_tiles, (
            f"The image of size {self.img_size} can't contain more than {(self.img_size // self.tile_size)**2} tiles."
            f"Choose a number of tiles 0 < n_tiles <= {(self.img_size // self.tile_size)**2} "
        )

    def set_tmp_path(self) -> None:
        self.tmp_path: Path = self.tmp_path / PandaTilesDataset._RELATIVE_ROOT_FOLDER
        os.makedirs(self.tmp_path, exist_ok=True)

    def create_mock_metadata_dataframe(self) -> pd.DataFrame:
        """Create a mock dataframe with random metadata."""
        csv_columns = [
            PandaTilesDataset.SLIDE_ID_COLUMN,
            PandaTilesDataset.TILE_ID_COLUMN,
            PandaTilesDataset.IMAGE_COLUMN,
            self.MASK_COLUMN,
            PandaTilesDataset.TILE_X_COLUMN,
            PandaTilesDataset.TILE_Y_COLUMN,
            self.OCCUPANCY,
            self.DATA_PROVIDER,
            self.ISUP_GRADE,
            self.GLEASON_SCORE,
        ]
        mock_metadata: dict = {col: [] for col in csv_columns}

        n_tiles_side = self.img_size // self.tile_size
        total_n_tiles = n_tiles_side ** 2
        tiles_count = 0

        # This is to make sure that the dataset contains at least one sample from each isup grade class.
        isup_grades = np.tile(list(self.ISUP_GRADE_MAPPING.keys()), self.n_slides // PandaTilesDataset.N_CLASSES + 1,)

        for slide_id in range(self.n_slides):

            data_provider = np.random.choice(self.DATA_PROVIDERS_VALUES)
            isup_grade = isup_grades[slide_id]
            gleason_score = np.random.choice(self.ISUP_GRADE_MAPPING[isup_grade])

            # pick a random n_tiles for each slide without exceeding the max n_tiles allowed
            max_n_tiles = (self.img_size // self.tile_size) ** 2
            n_tiles: int = min(np.random.randint(self.n_tiles // 2 + 1, 3 * self.n_tiles // 2), max_n_tiles)

            tiles_count += n_tiles
            coords = [
                (k // n_tiles_side, k % n_tiles_side)
                for k in np.random.choice(total_n_tiles, size=n_tiles, replace=False)
            ]

            for tile_id in range(n_tiles):
                tile_x = coords[tile_id][0] * self.tile_size
                tile_y = coords[tile_id][1] * self.tile_size
                mock_metadata[PandaTilesDataset.SLIDE_ID_COLUMN].append(f"_{slide_id}")
                mock_metadata[PandaTilesDataset.TILE_ID_COLUMN].append(f"_{slide_id}.{tile_x}x_{tile_y}y")
                mock_metadata[PandaTilesDataset.IMAGE_COLUMN].append(
                    f"_{slide_id}/train_images/{tile_x}x_{tile_y}y.png"
                )
                mock_metadata[self.MASK_COLUMN].append(f"_{slide_id}/train_label_masks/{tile_x}x_{tile_y}y_mask.png")
                mock_metadata[PandaTilesDataset.TILE_X_COLUMN].append(tile_x)
                mock_metadata[PandaTilesDataset.TILE_Y_COLUMN].append(tile_y)
                mock_metadata[self.OCCUPANCY].append(1.0)
                mock_metadata[self.DATA_PROVIDER].append(data_provider)
                mock_metadata[self.ISUP_GRADE].append(isup_grade)
                mock_metadata[self.GLEASON_SCORE].append(gleason_score)

        df = pd.DataFrame(data=mock_metadata)
        df.to_csv(os.path.join(self.tmp_path, PandaTilesDataset.DEFAULT_CSV_FILENAME), index=False)
        self.total_tiles: int = tiles_count
        return df

    def generate_mock_histo_data(self) -> None:
        # we retrieve all tiles at once, n_tiles updated l.100 to be used as a global batch_size for the dataloader
        # to be able to create slide with different number of tiles.
        tiles, _ = next(iter(self.dataloader)) if self.dataloader else (None, None)
        for i, row in self.dataframe.iterrows():
            slide_dir = self.tmp_path / f"{row[PandaTilesDataset.SLIDE_ID_COLUMN]}/train_images"
            mask_dir = self.tmp_path / f"{row[PandaTilesDataset.SLIDE_ID_COLUMN]}/train_label_masks"
            os.makedirs(slide_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            if self.mock_type == MockHistoDataType.PATHMNIST:
                tile = tiles[i] * 255
            elif self.mock_type == MockHistoDataType.FAKE:
                tile = torch.full(
                    fill_value=np.random.uniform(0, 255), size=(self.n_channels, self.tile_size, self.tile_size)
                )
            else:
                raise NotImplementedError

            save_image(tile, str(self.tmp_path / row[PandaTilesDataset.IMAGE_COLUMN]))
            random_mask = torch.randint(0, 256, size=(self.n_channels, self.tile_size, self.tile_size))
            save_image(random_mask.float(), str(self.tmp_path / row[self.MASK_COLUMN]))
