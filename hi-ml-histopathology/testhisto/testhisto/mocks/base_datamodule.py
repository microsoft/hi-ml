#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import medmnist
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from enum import Enum
from pathlib import Path
from medmnist import INFO
from typing import Optional
from torch.utils.data import DataLoader


class MockHistoDataType(Enum):
    PATHMNIST = "pathmnist"
    FAKE = "fake"


class MockHistoDataGenerator:
    """Base class for mock histo data generation.

    :param SLIDE_ID_COLUMN: CSV column name for slide id.
    :param METADATA_POSSIBLE_VALUES: Possible values to be assigned to the dataset metadata.
        The isup grades correspond to the gleason scores in the given order.
    :param METADATA_COLUMNS: Column names for all the metadata available on the CSV dataset file.
    :param N_GLEASON_SCORES: The number of possible gleason_scores.
    :param N_DATA_PROVIDERS: The number of possible data_providers.
    :param N_CLASSES: The number of possible isup_grades.
    """

    SLIDE_ID_COLUMN = "slide_id"
    METADATA_POSSIBLE_VALUES: dict = {
        "data_provider": ["site_0", "site_1"],
        "isup_grade": [0, 4, 1, 3, 0, 5, 2, 5, 5, 4, 4],
        "gleason_score": ["0+0", "4+4", "3+3", "4+3", "negative", "4+5", "3+4", "5+4", "5+5", "5+3", "3+5"],
    }
    METADATA_COLUMNS = tuple(METADATA_POSSIBLE_VALUES.keys())
    N_GLEASON_SCORES = len(METADATA_POSSIBLE_VALUES["gleason_score"])
    N_DATA_PROVIDERS = len(METADATA_POSSIBLE_VALUES["data_provider"])
    N_CLASSES = 6

    def __init__(
        self,
        tmp_path: Path,
        mock_type: MockHistoDataType = MockHistoDataType.PATHMNIST,
        seed: int = 42,
        n_tiles: int = 1,
        n_slides: int = 4,
        n_channels: int = 3,
        tile_size: int = 28,
    ) -> None:
        """
        :param tmp_path: A temporary directory to store all generated data.
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
        self.tmp_path = tmp_path
        self.mock_type = mock_type
        self.n_tiles = n_tiles
        self.n_slides = n_slides
        self.n_channels = n_channels
        self.tile_size = tile_size

        self.dataframe = self.create_mock_metadata_dataframe()
        self.dataloader = self.get_dataloader()

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

    def _get_pathmnist_dataloader(self) -> DataLoader:
        """Get a dataloader for pathmnist dataset. It returns tiles of shape (batch_size, 3, 28, 28).
        :return: A dataloader to sample pathmnist tiles.
        """
        info = INFO["pathmnist"]
        DataClass = getattr(medmnist, info["python_class"])
        data_transform = transforms.Compose([transforms.ToTensor()])
        dataset = DataClass(split="train", transform=data_transform, download=True)
        return DataLoader(dataset=dataset, batch_size=self.n_tiles, shuffle=True)

    def generate_mock_histo_data(self) -> None:
        """Create mock histo data and save it in the corresponding format: tiff for wsi and png for tiles"""
        raise NotImplementedError
