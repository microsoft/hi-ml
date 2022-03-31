#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import medmnist
import pandas as pd
import numpy as np

import torch.utils.data as data
import torchvision.transforms as transforms

from torch import Tensor
from medmnist import INFO
from tifffile import TiffWriter

from typing import Tuple, Optional, Iterable, List
from histopathology.datamodules.base_module import SlidesDataModule
from histopathology.datasets.base_dataset import SlidesDataset
from health_azure.utils import PathOrString

METADATA_POSSIBLE_VALUES = {
    "data_provider": ["site_0", "site_1"],
    "isup_grade": [0, 4, 1, 3, 0, 5, 2, 5, 5, 4, 4],
    "gleason_score": ["0+0", "4+4", "3+3", "4+3", "negative", "4+5", "3+4", "5+4", "5+5", "5+3", "3+5"],
}
N_GLEASON_SCORES = 11


class MockSlidesDataset(SlidesDataset):
    """Mock and child class of SlidesDataset, to be used for testing purposes.
    It overrides the following, according to the PANDA cohort setting:

    :param SLIDE_ID_COLUMN: CSV column name for slide ID set to "image_id".
    :param LABEL_COLUMN: CSV column name for tile label set to "isup_grade".
    :param METADATA_COLUMNS: Column names for all the metadata available on the CSV dataset file.
    """

    SLIDE_ID_COLUMN = "image_id"
    LABEL_COLUMN = "isup_grade"

    METADATA_COLUMNS = ("data_provider", "isup_grade", "gleason_score")

    def __init__(
        self, root: PathOrString, dataset_csv: Optional[PathOrString] = None, dataset_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        :param root: Root directory of the dataset.
        :param dataset_csv: Full path to a dataset CSV file, containing at least
        `TILE_ID_COLUMN`, `SLIDE_ID_COLUMN`, and `IMAGE_COLUMN`. If omitted, the CSV will be read
        from `"{root}/{DEFAULT_CSV_FILENAME}"`.
        :param dataset_df: A potentially pre-processed dataframe in the same format as would be read
        from the dataset CSV file, e.g. after some filtering. If given, overrides `dataset_csv`.
        """
        super().__init__(root, dataset_csv, dataset_df, validate_columns=False)
        slide_ids = self.dataset_df.index
        self.dataset_df[self.IMAGE_COLUMN] = slide_ids + ".tiff"
        self.validate_columns()


class MockSlidesDataModule(SlidesDataModule):
    """Mock and child class of SlidesDataModule. It overrides get_splits so that it uses MockSlidesDataset.
    """

    def get_splits(self) -> Tuple[MockSlidesDataset, MockSlidesDataset, MockSlidesDataset]:
        return (MockSlidesDataset(self.root_path), MockSlidesDataset(self.root_path), MockSlidesDataset(self.root_path))


def create_mock_metadata_dataframe(tmp_path: str, n_samples: int = 4, seed: int = 42) -> None:
    """Create a mock dataframe with random metadata.

    :param tmp_path: A temporary directory to store the mock CSV file.
    :param n_samples: Number of random samples to generate, defaults to 4.
    :param seed: pseudorandom number generator seed to use for mocking random metadata, defaults to 42.
    """
    np.random.seed(seed)
    mock_metadata: dict = {col: [] for col in [MockSlidesDataset.IMAGE_COLUMN, *MockSlidesDataset.METADATA_COLUMNS]}
    for i in range(n_samples):
        mock_metadata[MockSlidesDataset.IMAGE_COLUMN].append(f"_{i}")
        rand_id = np.random.randint(0, N_GLEASON_SCORES)
        for key, val in METADATA_POSSIBLE_VALUES:
            i = rand_id if len(val) == N_GLEASON_SCORES else np.random.randint(2)
            # We want to make sure we're picking the same random index (rand_id) for isup_grade and gleason_score.
            # Otherewise chose either site_0 or site_1 data_provider.
            mock_metadata[key].append(val[i])
    df = pd.DataFrame(data=mock_metadata)
    df.to_csv(os.path.join(tmp_path, MockSlidesDataset.DEFAULT_CSV_FILENAME), index=False)


def save_mock_wsi_as_tiff_file(file_name: str, levels: List[np.ndarray]) -> None:
    with TiffWriter(file_name, bigtiff=True) as tif:
        options = dict(photometric="rgb", compression="zlib")
        for i, serie in enumerate(levels):
            tif.write(serie, **options, subfiletype=int(i > 0))


def create_pathmnist_stitched_tiles(
    tiles: Tensor, sample_counter: int, img_size: int, n_channels: int, step_size: int, different_tiles: bool = False
) -> np.ndarray:
    mock_image = np.full(shape=(n_channels, img_size, img_size), fill_value=255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        tile = tiles[0] if not different_tiles else tile
        _tile = (tile.numpy() * 255).astype(np.uint8)
        mock_image[:, step_size * i : step_size * (i + 1), step_size * i : step_size * (i + 1)] = np.tile(_tile, (2, 2))
        if different_tiles:
            np.save(os.path.join("pathmnist", f"_{sample_counter}", f"tile_{i}.npy"), _tile)
        elif i == 0:
            np.save(os.path.join("pathmnist", f"_{sample_counter}_tile.npy"), _tile)
    return np.transpose(mock_image, (1, 2, 0))


def create_multi_resolution_wsi(mock_image: np.ndarray, n_levels: int) -> List[np.ndarray]:
    levels = [mock_image[:: 2 ** i, :: 2 ** i] for i in range(n_levels)]
    return levels


def get_pathmnist_data_loader(batch_size: int = 4) -> Iterable[Tensor]:
    info = INFO["pathmnist"]
    DataClass = getattr(medmnist, info["python_class"])
    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset = DataClass(split="train", transform=data_transform, download=True)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def create_pathmnist_mock_wsis(
    tile_size: int = 28,
    n_tiles: int = 2,
    n_repeat: int = 4,
    n_channels: int = 3,
    n_samples: int = 4,
    n_levels: int = 3,
    different_tiles: bool = False,
) -> None:

    data_loader = get_pathmnist_data_loader(n_repeat)
    for sample_counter in range(n_samples):
        if different_tiles:
            os.makedirs(f"pathmnist/_{sample_counter}", exist_ok=True)
        tiles, _ = next(iter(data_loader))
        mock_image = create_pathmnist_stitched_tiles(
            tiles,
            sample_counter,
            img_size=n_tiles * n_repeat * tile_size,
            n_channels=n_channels,
            step_size=n_tiles * tile_size,
            different_tiles=different_tiles,
        )
        levels = create_multi_resolution_wsi(mock_image, n_levels)
        save_mock_wsi_as_tiff_file(os.path.join("pathmnist", f"_{sample_counter}.tiff"), levels)


def create_fake_stitched_tiles(
    img_size: int, n_channels: int, step_size: int, n_repeat: int, fill_val: int
) -> np.ndarray:
    mock_image = np.full(shape=(n_channels, img_size, img_size), fill_value=1, dtype=np.uint8)
    for i in range(n_repeat):
        mock_image[:, step_size * i : step_size * (i + 1), step_size * i : step_size * (i + 1)] = fill_val * (i + 1)
    return np.transpose(mock_image, (1, 2, 0))


def create_fake_mock_wsis(
    tile_size: int = 28,
    n_tiles: int = 2,
    n_repeat: int = 4,
    n_channels: int = 3,
    n_samples: int = 4,
    n_levels: int = 3,
) -> None:

    for sample_counter in range(n_samples):
        mock_image = create_fake_stitched_tiles(
            img_size=n_tiles * n_repeat * tile_size,
            n_channels=n_channels,
            step_size=n_tiles * tile_size,
            n_repeat=n_repeat,
            fill_val=np.random.randint(0, 60),
        )
        levels = create_multi_resolution_wsi(mock_image, n_levels)
        save_mock_wsi_as_tiff_file(os.path.join("fake", f"_{sample_counter}.tiff"), levels)
