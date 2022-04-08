#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
import py
import numpy as np
import pandas as pd

from torch import Tensor
from tifffile import TiffWriter

from typing import Any, Tuple, Optional, List, Union
from histopathology.datamodules.base_module import SlidesDataModule
from histopathology.datasets.base_dataset import SlidesDataset
from health_azure.utils import PathOrString
from testhisto.utils.utils_base_datamodule import MockHistoDataGenerator, MockHistoDataType


class MockSlidesDataset(SlidesDataset):
    """Mock and child class of SlidesDataset, to be used for testing purposes.
    It overrides the following, according to the PANDA cohort settings:

    :param LABEL_COLUMN: CSV column name for tile label set to "isup_grade".
    :param METADATA_COLUMNS: Column names for all the metadata available on the CSV dataset file.
    """

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
    """Mock and child class of SlidesDataModule, overrides get_splits so that it uses MockSlidesDataset."""

    def get_splits(self) -> Tuple[MockSlidesDataset, MockSlidesDataset, MockSlidesDataset]:
        return tuple(MockSlidesDataset(self.root_path) for _ in range(3))


class MockWSIGenerator(MockHistoDataGenerator):
    """Generator class to create mock WSI on the fly. A mock WSI resembles to:
                                [**      ]
                                [  **    ]
                                [    **  ]
                                [      **]
        where * represents 2 tiles stitched along the Y axis.

    :param SLIDE_ID_COLUMN: CSV column name for slide id.
    :param DEFAULT_CSV_FILENAME: Default name of the dataset CSV at the dataset root directory.
    """

    SLIDE_ID_COLUMN = MockSlidesDataset.SLIDE_ID_COLUMN
    DEFAULT_CSV_FILENAME = MockSlidesDataset.DEFAULT_CSV_FILENAME

    def __init__(
        self,
        n_levels: int = 3,
        n_repeat_diag: int = 4,
        n_repeat_tile: int = 2,
        background_val: Union[int, float] = 255,
        **kwargs: Any,
    ) -> None:
        """
        :param n_levels: Number of levels for multi resolution WSI.
        :param n_repeat_diag: Number of repeat time along the diagonal axis, defaults to 4.
        :param n_repeat_tile: Number of repeat times of a tile along both Y and X axes, defaults to 2.
        :param background_val: A value to assign to the background, defaults to 255.
        """
        super().__init__(**kwargs)

        self.n_levels = n_levels
        self.n_repeat_diag = n_repeat_diag
        self.n_repeat_tile = n_repeat_tile
        self.background_val = background_val

        self.step_size = self.tile_size * self.n_repeat_tile  # the step_size represents the diagonal square size.
        self._dtype = np.uint8 if type(background_val) == int else np.float32
        self.img_size: int = self.n_repeat_diag * self.n_repeat_tile * self.tile_size

    def create_mock_metadata_dataframe(self) -> pd.DataFrame:
        """Create a mock dataframe with random metadata."""
        mock_metadata: dict = {col: [] for col in [self.SLIDE_ID_COLUMN, *self.METADATA_COLUMNS]}
        for i in range(self.n_slides):
            mock_metadata[MockSlidesDataset.SLIDE_ID_COLUMN].append(f"_{i}")
            rand_id = np.random.randint(0, self.N_GLEASON_SCORES)
            for key, val in self.METADATA_POSSIBLE_VALUES.items():
                i = rand_id if len(val) == self.N_GLEASON_SCORES else np.random.randint(self.N_DATA_PROVIDERS)
                # Make sure to pick the same random index (rand_id) for isup_grade and gleason_score, otherwise
                # chose among possible data_providers.
                mock_metadata[key].append(val[i])
        df = pd.DataFrame(data=mock_metadata)
        df.to_csv(os.path.join(self.tmp_path, self.DEFAULT_CSV_FILENAME), index=False)
        return df

    def _create_wsi_from_stitched_tiles(self, tiles: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Create a whole slide image by stitching tiles along the diagonal axis.

        :param tiles: A tensor of tiles of shape (n_tiles, n_channels, tile_size, tile_size).
        :return: returns a wsi of shape (img_size, img_size, n_channels) and the tiles used to create it.
        The image is  in channels_last format so that it can save by a TiffWriter.
        """
        mock_image = np.full(
            shape=(self.n_channels, self.img_size, self.img_size), fill_value=self.background_val, dtype=self._dtype
        )
        dump_tiles = []
        for i in range(self.n_repeat_diag):
            if self.mock_type == MockHistoDataType.PATHMNIST:
                if i == 0 or self.n_tiles > 1:
                    tile = (
                        (tiles[i % self.n_tiles].numpy() * 255).astype(self._dtype)
                        if self._dtype == np.uint8
                        else tiles[i % self.n_tiles].numpy()
                    )
                    # fill the square diagonal with tile repeated n_repeat_tile times along X and Y axis.
                    fill_square = np.tile(tile, (self.n_repeat_tile, self.n_repeat_tile))
                    dump_tiles.append(tile)

            elif self.mock_type == MockHistoDataType.FAKE:
                if i == 0 or self.n_tiles > 1:
                    # pick a random fake value to fill in the square diagonal.
                    fill_square = np.random.uniform(0, self.background_val / (self.n_repeat_diag + 1) * (i + 1))
                    dump_tiles.append(
                        np.full(
                            shape=(self.n_channels, self.tile_size, self.tile_size),
                            fill_value=fill_square,
                            dtype=self._dtype,
                        )
                    )
            else:
                raise NotImplementedError
            mock_image[
                :, self.step_size * i : self.step_size * (i + 1), self.step_size * i : self.step_size * (i + 1)
            ] = fill_square
        return np.transpose(mock_image, (1, 2, 0)), np.array(dump_tiles)  # switch to channels_last.

    @staticmethod
    def _save_mock_wsi_as_tiff_file(file_path: Union[py.path.local, Path], wsi_levels: List[np.ndarray]) -> None:
        """Save a mock whole slide image as a tiff file of pyramidal levels.
        Warning: this function expects images to be in channels_last format (H, W, C).

        :param file_name: The tiff file name path.
        :param wsi_levels: List of whole slide images of different resolution levels in channels_last format.
        """
        with TiffWriter(file_path, bigtiff=True) as tif:
            options = dict(photometric="rgb", compression="zlib")
            for i, wsi_level in enumerate(wsi_levels):
                # the subfiletype parameter is a bitfield that determines if the wsi_level is a reduced version of
                # another image.
                tif.write(wsi_level, **options, subfiletype=int(i > 0))

    def _create_multi_resolution_wsi(self, mock_image: np.ndarray) -> List[np.ndarray]:
        """Create multi resolution versions of a mock image via 2 factor downsampling.

        :param mock_image: A mock image in channels_last format (H, W, 3).
        :return: Returns a list of n_levels downsampled versions of the original mock image.
        """
        levels = [mock_image[:: 2 ** i, :: 2 ** i] for i in range(self.n_levels)]
        return levels

    def generate_mock_histo_data(self) -> None:
        """Create mock wsi and save them as tiff files"""
        for sample_counter in range(self.n_slides):
            tiles, _ = next(iter(self.dataloader)) if self.dataloader else None, None
            mock_image, dump_tiles = self._create_wsi_from_stitched_tiles(tiles[0])
            wsi_levels = self._create_multi_resolution_wsi(mock_image)
            self._save_mock_wsi_as_tiff_file(self.tmp_path / f"_{sample_counter}.tiff", wsi_levels)
            np.save(str(self.tmp_path / f"_{sample_counter}_tile.npy"), dump_tiles)
