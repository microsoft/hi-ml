#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum
import os
from pathlib import Path
import numpy as np
import pandas as pd

from torch import Tensor
from tifffile import TiffWriter

from typing import Any, Tuple, List, Union
from histopathology.datasets.panda_dataset import PandaDataset
from testhisto.mocks.base_data_generator import MockHistoDataGenerator, MockHistoDataType


class TilesPositioningType(Enum):
    DIAGONAL = 0
    RANDOM = 1


class MockPandaSlidesGenerator(MockHistoDataGenerator):
    """Generator class to create mock WSI on the fly. A mock WSI resembles to:
                                [**      ]
                                [  **    ]
                                [    **  ]
                                [      **]
        where * represents 2 tiles stitched along the Y axis, if tiles positioning is diagonal.
        Tiles are positioned randomly on the WSI grid whem tiles positioning type is random.
    """

    ISUP_GRADE = "isup_grade"

    def __init__(
        self,
        n_levels: int = 3,
        n_repeat_diag: int = 4,
        n_repeat_tile: int = 2,
        background_val: Union[int, float] = 255,
        tiles_pos_type: TilesPositioningType = TilesPositioningType.DIAGONAL,
        **kwargs: Any,
    ) -> None:
        """
        :param n_levels: Number of levels for multi resolution WSI.
        :param n_repeat_diag: Number of repeat time along the diagonal axis, defaults to 4.
        :param n_repeat_tile: Number of repeat times of a tile along both Y and X axes, defaults to 2.
        :param background_val: A value to assign to the background, defaults to 255.
        :param tiles_pos_type: The tiles positioning type to define how tiles should be positioned within the WSI grid,
        defaults to TilesPositioningType.DIAGONAL.
        :param kwargs: Same params passed to MockHistoDataGenerator.
        """
        super().__init__(**kwargs)

        self.n_levels = n_levels
        self.n_repeat_diag = n_repeat_diag
        self.n_repeat_tile = n_repeat_tile
        self.background_val = background_val
        self.tiles_pos_type = tiles_pos_type

        self.step_size = self.tile_size * self.n_repeat_tile
        self._dtype = np.uint8 if type(background_val) == int else np.float32
        self.img_size: int = self.n_repeat_diag * self.n_repeat_tile * self.tile_size

    def create_mock_metadata_dataframe(self) -> pd.DataFrame:
        """Create a mock dataframe with random metadata."""
        isup_grades = np.tile(list(self.ISUP_GRADE_MAPPING.keys()), self.n_slides // PandaDataset.N_CLASSES + 1,)
        mock_metadata: dict = {col: [] for col in [PandaDataset.SLIDE_ID_COLUMN, *PandaDataset.METADATA_COLUMNS]}
        for slide_id in range(self.n_slides):
            mock_metadata[PandaDataset.SLIDE_ID_COLUMN].append(f"_{slide_id}")
            mock_metadata[self.DATA_PROVIDER].append(np.random.choice(self.DATA_PROVIDERS_VALUES))
            mock_metadata[self.ISUP_GRADE].append(isup_grades[slide_id])
            mock_metadata[self.GLEASON_SCORE].append(np.random.choice(self.ISUP_GRADE_MAPPING[isup_grades[slide_id]]))
        df = pd.DataFrame(data=mock_metadata)
        df.to_csv(self.tmp_path / PandaDataset.DEFAULT_CSV_FILENAME, index=False)
        return df

    def create_mock_wsi(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.tiles_pos_type == TilesPositioningType.DIAGONAL:
            return self._create_wsi_from_stitched_tiles()
        elif self.tiles_pos_type == TilesPositioningType.RANDOM:
            return self._create_wsi_from_randomly_positioned_tiles(), None
        else:
            raise NotImplementedError

    def _create_wsi_from_stitched_tiles(self, tiles: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Create a whole slide image by stitching tiles along the diagonal axis.

        :param tiles: A tensor of tiles of shape (n_tiles, n_channels, tile_size, tile_size).
        :return: returns a wsi of shape (img_size, img_size, n_channels) and the tiles used to create it.
        The image is  in channels_last format so that it can save by TiffWriter.
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
                :, self.tile * i: self.step_size * (i + 1), self.step_size * i: self.step_size * (i + 1)
            ] = fill_square
        return np.transpose(mock_image, (1, 2, 0)), np.array(dump_tiles)  # switch to channels_last.

    def _create_wsi_from_randomly_positioned_tiles(self, tiles: Tensor) -> np.ndarray:
        """Create a whole slide image by positioning tiles randomly in the whole slide image grid.

        :param tiles: A tensor of tiles of shape (n_tiles, n_channels, tile_size, tile_size).
        :return: returns a wsi of shape (img_size, img_size, n_channels) in channels_last format so that it can save by 
        TiffWriter.
        """
        mock_image = np.full(
            shape=(self.n_channels, self.img_size, self.img_size), fill_value=self.background_val, dtype=self._dtype
        )

        n_tiles_side = self.img_size // self.tile_size
        total_n_tiles = n_tiles_side ** 2

        # pick a random n_tiles for each slide
        n_tiles = np.random.randint(self.n_tiles // 2 + 1, 3 * self.n_tiles // 2)
        coords = [
            (k // n_tiles_side, k % n_tiles_side) for k in np.random.choice(total_n_tiles, size=n_tiles, replace=False)
        ]
        for i, tile in enumerate(tiles):
            x, y = coords[i][0], coords[i][1]
            mock_image[:, x: x + self.tile_size, y: y + self.tile_size] = tile
            

    @staticmethod
    def _save_mock_wsi_as_tiff_file(file_path: Path, wsi_levels: List[np.ndarray]) -> None:
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
        iterator = iter(self.dataloader) if self.dataloader else None
        os.makedirs(self.tmp_path / "train_images", exist_ok=True)
        os.makedirs(self.tmp_path / "dump_tiles", exist_ok=True)
        for slide_counter in range(self.n_slides):
            tiles, _ = next(iterator) if iterator else (None, None)
            mock_image, dump_tiles = self.create_mock_wsi(tiles)
            wsi_levels = self._create_multi_resolution_wsi(mock_image)
            self._save_mock_wsi_as_tiff_file(self.tmp_path / "train_images" / f"_{slide_counter}.tiff", wsi_levels)
            np.save(self.tmp_path / "dump_tiles" / f"_{slide_counter}.npy", dump_tiles)
