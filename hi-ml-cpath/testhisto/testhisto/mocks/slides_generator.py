#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import torch
from tifffile.tifffile import TiffWriter, PHOTOMETRIC, COMPRESSION
from torch import Tensor
from health_cpath.datasets.panda_dataset import PandaDataset
from health_cpath.preprocessing.tiff_conversion import ResolutionUnit
from testhisto.mocks.base_data_generator import MockHistoDataGenerator, MockHistoDataType, PANDA_N_CLASSES


class TilesPositioningType(Enum):
    DIAGONAL = 0
    RANDOM = 1


class MockPandaSlidesGenerator(MockHistoDataGenerator):
    """Generator class to create mock WSI on the fly.
        If tiles positioning is diagonal, a mock WSI resembles to:
                                [**      ]
                                [  **    ]
                                [    **  ]
                                [      **]
        where * represents 2 tiles stitched along the Y axis.
        If tiles positioning is random, tiles are positioned randomly on the WSI grid.
    """

    ISUP_GRADE = "isup_grade"

    def __init__(
        self,
        n_levels: int = 3,
        n_repeat_diag: int = 4,
        n_repeat_tile: int = 2,
        background_val: Union[int, float] = 255,
        tiles_pos_type: TilesPositioningType = TilesPositioningType.DIAGONAL,
        n_tiles_list: Optional[List[int]] = None,
        resultion_unit: Optional[ResolutionUnit] = ResolutionUnit.CENTIMETER,
        **kwargs: Any,
    ) -> None:
        """
        :param n_levels: Number of levels for multi resolution WSI.
        :param n_repeat_diag: Number of repeat time along the diagonal axis, defaults to 4.
        :param n_repeat_tile: Number of repeat times of a tile along both Y and X axes, defaults to 2.
        :param background_val: A value to assign to the background, defaults to 255.
        :param tiles_pos_type: The tiles positioning type to define how tiles should be positioned within the WSI grid,
        defaults to `TilesPositioningType.DIAGONAL`.
        :param n_tiles_list: A list to use different n_tiles per slide for randomly positioned tiles.
        :param resultion_unit: The resolution unit to use for writing the WSI, defaults to `ResolutionUnit.CENTIMETER`.
        :param kwargs: Same params passed to MockHistoDataGenerator.
        """
        self.generated_files: List[str] = []
        super().__init__(**kwargs)

        self.n_levels = n_levels
        self.n_repeat_diag = n_repeat_diag
        self.n_repeat_tile = n_repeat_tile
        self.background_val = background_val
        self.tiles_pos_type = tiles_pos_type

        self.step_size = self.tile_size * self.n_repeat_tile
        self._dtype = np.uint8 if type(background_val) == int else np.float32
        self.img_size: int = self.n_repeat_diag * self.n_repeat_tile * self.tile_size
        self.n_tiles_list = n_tiles_list
        self.resolution_unit = resultion_unit

        if self.n_tiles_list:
            assert len(self.n_tiles_list) == self.n_slides, "n_tiles_list length should be equal to n_slides"
            assert self.tiles_pos_type == TilesPositioningType.RANDOM, "different n_tiles enabled only for randomly "
            "positionned tiles."

    def validate(self) -> None:
        assert (
            self.n_slides >= PANDA_N_CLASSES
        ), f"The number of slides should be >= PANDA_N_CLASSES (i.e., {PANDA_N_CLASSES})"

    def create_mock_metadata_dataframe(self) -> pd.DataFrame:
        """Create a mock dataframe with random metadata."""
        isup_grades = np.tile(list(self.ISUP_GRADE_MAPPING.keys()), self.n_slides // PANDA_N_CLASSES + 1,)
        mock_metadata: dict = {
            col: [] for col in [PandaDataset.SLIDE_ID_COLUMN, PandaDataset.MASK_COLUMN, *PandaDataset.METADATA_COLUMNS]
        }
        for slide_id in range(self.n_slides):
            mock_metadata[PandaDataset.SLIDE_ID_COLUMN].append(f"_{slide_id}")
            mock_metadata[PandaDataset.MASK_COLUMN].append(f"_{slide_id}_mask")
            mock_metadata[self.DATA_PROVIDER].append(np.random.choice(self.DATA_PROVIDERS_VALUES))
            mock_metadata[self.ISUP_GRADE].append(isup_grades[slide_id])
            mock_metadata[self.GLEASON_SCORE].append(np.random.choice(self.ISUP_GRADE_MAPPING[isup_grades[slide_id]]))
        df = pd.DataFrame(data=mock_metadata)
        csv_filename = self.dest_data_path / PandaDataset.DEFAULT_CSV_FILENAME
        df.to_csv(csv_filename, index=False)

    def create_mock_wsi(self, tiles: Tensor) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.tiles_pos_type == TilesPositioningType.DIAGONAL:
            return self._create_wsi_from_stitched_tiles_along_diagonal_axis(tiles)
        elif self.tiles_pos_type == TilesPositioningType.RANDOM:
            return self._create_wsi_from_randomly_positioned_tiles(tiles), None
        else:
            raise NotImplementedError

    def _create_wsi_from_stitched_tiles_along_diagonal_axis(self, tiles: Tensor) -> Tuple[np.ndarray, np.ndarray]:
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
                        (tiles[i % self.n_tiles].numpy()).astype(self._dtype)
                        if self._dtype == np.uint8
                        else tiles[i % self.n_tiles].numpy()
                    )
                    # fill the square diagonal with tile repeated n_repeat_tile times along X and Y axis.
                    fill_square: Union[np.ndarray, float] = np.tile(tile, (self.n_repeat_tile, self.n_repeat_tile))
                    dump_tiles.append(tile)

            elif self.mock_type == MockHistoDataType.FAKE:
                if i == 0 or self.n_tiles > 1:
                    # pick a random fake value to fill in the square diagonal.
                    upper = self.background_val / (self.n_repeat_diag + 1) * (i + 1)
                    fill_square = np.random.uniform(0, upper)
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
                :, self.step_size * i: self.step_size * (i + 1), self.step_size * i: self.step_size * (i + 1)
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
        coords = [
            (k // n_tiles_side, k % n_tiles_side)
            for k in np.random.choice(total_n_tiles, size=self.n_tiles, replace=False)
        ]
        for i in range(self.n_tiles):
            x, y = self.tile_size * np.array(coords[i])
            if self.mock_type == MockHistoDataType.PATHMNIST:
                new_tile = tiles[i].numpy()
            elif self.mock_type == MockHistoDataType.FAKE:
                new_tile = np.random.uniform(0, self.background_val / (self.n_repeat_diag + 1) * (i + 1))
            else:
                raise NotImplementedError
            mock_image[:, x: x + self.tile_size, y: y + self.tile_size] = new_tile
        return np.transpose(mock_image, (1, 2, 0))

    def _save_mock_wsi_as_tiff_file(self, file_path: Path, wsi_levels: List[np.ndarray]) -> None:
        """Save a mock whole slide image as a tiff file of pyramidal levels.
        Warning: this function expects images to be in channels_last format (H, W, C).

        :param file_name: The tiff file name path.
        :param wsi_levels: List of whole slide images of different resolution levels in channels_last format.
        """
        with TiffWriter(file_path, bigtiff=True) as tif:
            options = dict(
                software='tifffile',
                metadata={'axes': 'YXC'},
                photometric=PHOTOMETRIC.RGB,
                resolutionunit=self.resolution_unit,
                compression=COMPRESSION.ADOBE_DEFLATE,  # ADOBE_DEFLATE aka ZLIB lossless compression
                tile=(16, 16),
            )
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

        slide_dir = self.dest_data_path / "train_images"
        slide_dir.mkdir(parents=True, exist_ok=True)
        tile_dir = self.dest_data_path / "dump_tiles"
        tile_dir.mkdir(parents=True, exist_ok=True)

        for slide_counter in range(self.n_slides):

            if self.n_tiles_list:
                self.total_tiles = self.n_tiles_list[slide_counter]
                self.n_tiles: int = self.n_tiles_list[slide_counter]
                self.dataloader: torch.utils.data.DataLoader = self.get_dataloader()
                iterator = iter(self.dataloader)

            tiles, _ = next(iterator) if iterator else (None, None)
            mock_image, dump_tiles = self.create_mock_wsi(tiles)
            wsi_levels = self._create_multi_resolution_wsi(mock_image)

            slide_tiff_filename = self.dest_data_path / "train_images" / f"_{slide_counter}.tiff"
            self._save_mock_wsi_as_tiff_file(slide_tiff_filename, wsi_levels)
            self.generated_files.append(str(slide_tiff_filename))

            if dump_tiles is not None:
                dump_tiles_filename = self.dest_data_path / "dump_tiles" / f"_{slide_counter}.npy"
                np.save(dump_tiles_filename, dump_tiles)
