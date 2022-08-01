#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import TensorDataset

from enum import Enum
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader


PANDA_N_CLASSES = 6


class MockHistoDataType(Enum):
    PATHMNIST = "PathMNIST"
    FAKE = "fake"


class MockHistoDataGenerator:
    """Base class for mock histo data generation.

    :param MASK_COLUMN: CSV column name for relative path to mask file.
    :param DATA_PROVIDER: CSV column name for data provider.
    :param ISUP_GRADE: CSV column name for isup grade.
    :param GLEASON_SCORE: CSV column name for gleason score.
    :param DATA_PROVIDERS_VALUES: Possible values to be assigned to data provider column. The values mapped to
    :param ISUP_GRADE_MAPPING: Possible values to be assigned to isup grade column. The values mapped to
    isup_grades are the possible gleason_scores.
    """

    MASK_COLUMN = "mask"
    OCCUPANCY = "occupancy"
    DATA_PROVIDER = "data_provider"
    ISUP_GRADE = "slide_isup_grade"
    GLEASON_SCORE = "gleason_score"

    DATA_PROVIDERS_VALUES = ["site_0", "site_1"]
    ISUP_GRADE_MAPPING = {
        0: ["0+0", "negative"],
        4: ["4+4", "5+3", "3+5"],
        1: ["3+3"],
        3: ["4+3"],
        2: ["3+4"],
        5: ["4+5", "5+4", "5+5"],
    }

    def __init__(
        self,
        dest_data_path: Path,
        src_data_path: Optional[Path] = None,
        mock_type: MockHistoDataType = MockHistoDataType.PATHMNIST,
        seed: int = 42,
        n_tiles: int = 1,
        n_slides: int = 4,
        n_channels: int = 3,
        tile_size: int = 28,
    ) -> None:
        """
        :param dest_data_path: A temporary directory to store all generated data.
        :param src_data_path: An optional path directory where the source dataset is stored (e.g., pathmnist dataset).
        :param mock_type: The wsi generator mock type. Supported mock types are:
            WSIMockType.PATHMNIST: for creating mock WSI by stitching tiles from pathmnist.
            WSIMockType.FAKE: for creating mock WSI by stitching fake tiles.
        :param seed: pseudorandom number generator seed to use for mocking random metadata, defaults to 42.
        :param n_tiles: how many tiles per slide to load from pathmnist dataloader, defaults to 1.
            if n_tiles > 1 WSIs are generated from different tiles in the subclass MockWSIGenerator.
        :param n_slides: Number of random slides to generate, defaults to 4.
        :param n_channels: Number of channels, defaults to 3.
        :param tile_size: The tile size, defaults to 28.
        """
        np.random.seed(seed)
        self.dest_data_path = dest_data_path
        self.src_data_path = src_data_path
        self.mock_type = mock_type
        self.n_tiles = n_tiles
        self.total_tiles = n_tiles
        self.n_slides = n_slides
        self.n_channels = n_channels
        self.tile_size = tile_size

        self.validate()
        self.dest_data_path.mkdir(parents=True, exist_ok=True)

        self.dataframe = self.create_mock_metadata_dataframe()
        self.dataloader = self.get_dataloader()

    def validate(self) -> None:
        pass

    def create_mock_metadata_dataframe(self) -> pd.DataFrame:
        """Create a mock dataframe with random metadata."""
        raise NotImplementedError

    def get_dataloader(self) -> Optional[DataLoader]:
        if self.mock_type == MockHistoDataType.PATHMNIST:
            return self._get_pathmnist_dataloader()
        elif self.mock_type == MockHistoDataType.FAKE:
            # we don't need a dataloader for MockHistoDataType.FAKE, a random fake value is picked (l.137 in MockWSI)
            return None
        else:
            raise NotImplementedError

    def generate_mock_histo_data(self) -> None:
        """Create mock histo data and save it in the corresponding format: tiff for wsi and png for tiles"""
        raise NotImplementedError

    def _create_pathmnist_dataset(self, split: str) -> TensorDataset:
        """Create pathmnist torch dataset from local copy of a dataset.

        :param split: The split subset. It takes values in ["train", "val", "test"]
        :return: A TensorDataset for pathmnist tiles.
        """
        assert split in ["train", "val", "test"], "Please choose a split string among [train, val, test]"
        assert self.src_data_path is not None  # for mypy
        npz_file = np.load(self.src_data_path / f"{self.mock_type.value.lower()}.npz")

        imgs = torch.Tensor(npz_file[f"{split}_images"]).permute(0, 3, 1, 2).int()
        labels = torch.Tensor(npz_file[f"{split}_labels"])

        return TensorDataset(imgs, labels)

    def _get_pathmnist_dataloader(self) -> DataLoader:
        """Get a dataloader for pathmnist dataset. It returns tiles of shape (self.n_tiles, 3, 28, 28).
        :return: A dataloader to sample pathmnist tiles.
        """
        return DataLoader(
            dataset=self._create_pathmnist_dataset(split="train"), batch_size=self.total_tiles, shuffle=True
        )
