#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import py
import medmnist
import numpy as np
import torchvision.transforms as transforms


from enum import Enum
from pathlib import Path
from medmnist import INFO
from typing import Optional, Union

from torch.utils.data import DataLoader


class MockWSIType(Enum):
    PATHMNIST = "pathmnist"
    FAKE = "fake"


class MockHistoGenerator:
    """Base class for mock histo data generation.

    :param METADATA_POSSIBLE_VALUES: Possible values to be assigned to the dataset metadata.
        The isup grades correspond to the gleason scores in the given order.
    :param N_GLEASON_SCORES: The number of possible gleason_scores.
    :param N_DATA_PROVIDERS: the number of possible data_providers.
    """

    METADATA_POSSIBLE_VALUES: dict = {
        "data_provider": ["site_0", "site_1"],
        "isup_grade": [0, 4, 1, 3, 0, 5, 2, 5, 5, 4, 4],
        "gleason_score": ["0+0", "4+4", "3+3", "4+3", "negative", "4+5", "3+4", "5+4", "5+5", "5+3", "3+5"],
    }
    N_GLEASON_SCORES = len(METADATA_POSSIBLE_VALUES["gleason_score"])
    N_DATA_PROVIDERS = len(METADATA_POSSIBLE_VALUES["data_provider"])

    def __init__(
        self,
        tmp_path: Union[py.path.local, Path],
        mock_type: MockWSIType = MockWSIType.PATHMNIST,
        seed: int = 42,
        n_tiles: int = 1,
        n_samples: int = 4,
        tile_size: int = 28,
    ) -> None:
        """
        :param tmp_path: A temporary directory to store all generated data.
        :param mock_type: The wsi generator mock type. Supported mock types are:
            WSIMockType.PATHMNIST: for creating mock WSI by stitching tiles from pathmnist.
            WSIMockType.FAKE: for creating mock WSI by stitching fake tiles.
        :param seed: pseudorandom number generator seed to use for mocking random metadata, defaults to 42.
        :param n_tiles: how many tiles per batch to load from pathmnist dataloader, defaults to 1.
            if n_tiles > 1 WSIs are generated from different tiles in the subclass MockWSIGenerator.
        :param n_samples: Number of random samples to generate, defaults to 4.
        :param tile_size: The tile size, defaults to 28.
        """
        np.random.seed(seed)
        self.tmp_path = tmp_path
        self.mock_type = mock_type
        self.n_tiles = n_tiles
        self.n_samples = n_samples
        self.tile_size = tile_size

        self.create_mock_metadata_dataframe()
        self.dataloader = self.get_dataloader()

    def create_mock_metadata_dataframe(self) -> None:
        """Create a mock dataframe with random metadata."""
        raise NotImplementedError

    def get_dataloader(self) -> Optional[DataLoader]:
        if self.mock_type == MockWSIType.PATHMNIST:
            return self._get_pathmnist_dataloader()
        elif self.mock_type == MockWSIType.FAKE:
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
