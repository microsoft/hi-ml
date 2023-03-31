#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
from enum import Enum
import logging
import math
import numpy as np

from health_cpath.preprocessing.loading import WSIBackend
from health_cpath.utils.naming import SlideKey
from monai.data.wsi_reader import WSIReader
from monai.transforms import MapTransform
from openslide import OpenSlide
from pathlib import Path
from tifffile.tifffile import TiffWriter, PHOTOMETRIC, COMPRESSION
from typing import Any, Dict, List, Optional, Tuple


AMPERSAND = "&"
UNDERSCORE = "_"
TIFF_EXTENSION = ".tiff"


class ResolutionUnit(str, Enum):
    """The unit of the resolution of the tiff file. This is used to calculate the resolution of the tiff file."""

    INCH = "inch"
    CENTIMETER = "centimeter"
    MILLIMETER = "millimeter"
    MICROMETER = "micrometer"


class ConvertWSIToTiffd(MapTransform):
    """Converts a wsi file to a tiff file. The tiff file is saved in the output_folder with the same name as the src
    file but with the tiff extension. Ampersands are replaced by the replace_ampersand_by string. The tiff file
    contains the image data at the target magnifications. If target_magnifications is None, the tiff file contains the
    image data at all magnifications. If add_lowest_magnification is True, the tiff file also contains the image data
    at the lowest magnification. The tiff file is saved with the compression specified by the compression parameter with
    a fixed tile size. This works with all supported wsi formats by openslide.
    """

    OBJECTIVE_POWER_KEY = "openslide.objective-power"
    RESOLUTION_UNIT_KEY = "tiff.ResolutionUnit"
    RESOLUTION_UNIT = ResolutionUnit.CENTIMETER
    SOFTWARE = "tifffile"

    def __init__(
        self,
        output_folder: Path,
        image_key: str = SlideKey.IMAGE,
        target_magnifications: Optional[List[float]] = [10.0],
        add_lowest_magnification: bool = False,
        default_base_objective_power: Optional[float] = None,
        replace_ampersand_by: str = UNDERSCORE,
        compression: COMPRESSION = COMPRESSION.ADOBE_DEFLATE,
        tile_size: int = 512,
        min_file_size: int = 0,
        verbose: bool = False,
    ) -> None:
        """
        :param output_folder: The directory where the tiff file will be saved.
        :param image_key: The key of the image that should be converted in the data dictionary, defaults to
            `SlideKey.IMAGE`.
        :param target_magnifications: The magnifications that should be read, converted and written to the output file,
            e.g. [10., 20.], defaults to [10.]. If target_magnifications is None, the tiff file will contain the image
            data at all magnifications.
        :param add_lowest_magnification: A flag indicating whether the tiff file should also contain the image data at
            the lowest magnification, defaults to False. This is useful if the lowest magnification of the wsi is not
            part of the target magnifications and one wants to use the lowest magnification for faster processing.
        :param default_base_objective_power: The base objective power of the wsi. This is used to calculate the
            magnification of the wsi. If the objective power is not found in the wsi properties, the
            base_objective_power is used instead. If the objective power is not found in the wsi properties and
            base_objective_power is None, an error is raised, defaults to None.
        :param replace_ampersand_by: A string that is used to replace ampersands in the src file name, defaults to
            `UNDERSCORE`. This is useful because ampersands in file names can cause problems in cloud storage.
        :param compression: The compression that is used to save the tiff file, defaults to `COMPRESSION.ADOBE_DEFLATE`
            aka ZLIB that is lossless compression. Make sure to use one of these options (RAW, LZW, JPEG, JPEG2000) so
            that the converted files are readable by cucim.
        :param tile_size: The size of the tiles that are used to write the tiff file, defaults to 512.
        :param min_file_size: The minimum size of the tiff file in bytes. If the tiff file is smaller than this size, it
            will get overwritten. Defaults to 0.
        :param verbose: A flag to enable verbose logging, defaults to False.
        """
        self.output_folder = output_folder
        self.image_key = image_key
        if target_magnifications is not None:
            target_magnifications.sort(reverse=True)
        self.target_magnifications = target_magnifications if target_magnifications else None
        self.add_lowest_magnification = add_lowest_magnification
        self.replace_ampersand_by = replace_ampersand_by
        self.default_base_objective_power = default_base_objective_power
        self.wsi_reader = WSIReader(backend=WSIBackend.OPENSLIDE)
        self.compression = compression
        self.tile_size = tile_size
        self.min_file_size = min_file_size
        self.verbose = verbose

    def get_tiff_path(self, src_path: Path) -> Path:
        """Returns the path to the tiff file that will be created from the src file. The tiff file is saved in the
        output_folder with the same name as the src file but with the tiff extension. Ampersands are replaced by the
        replace_ampersand_by string.

        :param src_path: The path to the src file.
        :return: The path to the tiff file that will be created from the src file.
        """
        tiff_filename = src_path.with_suffix(TIFF_EXTENSION).name
        tiff_filename = tiff_filename.replace(AMPERSAND, self.replace_ampersand_by)
        return self.output_folder / tiff_filename

    def _get_base_objective_power(self, wsi_obj: OpenSlide) -> float:
        """Returns the objective power of the wsi. The objective power is extracted from the wsi properties. If the
        objective power is not found in the wsi properties, the base_objective_power is used instead.

        :param wsi_obj: The wsi object in openslide format
        :raises ValueError: Raises an error if the objective power is not found in the wsi properties and
            base_objective_power is None
        :return: The base objective power of the wsi
        """
        base_objective_power = wsi_obj.properties.get(self.OBJECTIVE_POWER_KEY, self.default_base_objective_power)
        if base_objective_power is None:
            raise ValueError(
                f"Could not find {self.OBJECTIVE_POWER_KEY} in wsi properties. Please specify a default value for "
                "default_base_objective_power."
            )
        return float(base_objective_power)

    def _get_target_level(self, wsi_obj: OpenSlide, target_magnification: float) -> int:
        """Returns the level of the wsi pyramid that is equal to the target magnification.

        :param wsi_obj: the wsi object in openslide format
        :param target_magnification: the target magnification e.g. 10x
        """
        base_objective_power = self._get_base_objective_power(wsi_obj)
        for level_idx, level_downsample in enumerate(wsi_obj.level_downsamples):
            objective_power = base_objective_power / level_downsample
            if math.isclose(objective_power, target_magnification, rel_tol=1e-3):
                return level_idx
        raise ValueError(f"Target magnification {target_magnification} not found")

    def _get_highest_level(self, wsi_obj: OpenSlide) -> int:
        """Returns the highest resolution level of the wsi pyramid."""
        return len(wsi_obj.level_downsamples) - 1

    def get_target_levels(self, wsi_obj: Any) -> List[int]:
        """Returns the levels of the wsi pyramid that will be saved in the tiff file. If target_magnifications is not
        None, we return the levels that correspond to the target magnifications. If add_lowest_magnification is True, we
        add the highest resolution level of the wsi pyramid if it is not already in the target levels.
        Otherwise, we return all levels.

        :param wsi_obj: The wsi object in openslide format
        :return: A list of levels that will be saved in the tiff file
        """
        if self.target_magnifications is not None:
            target_levels = [self._get_target_level(wsi_obj, mag) for mag in self.target_magnifications]
            if self.add_lowest_magnification:
                highest_level = self._get_highest_level(wsi_obj)
                if highest_level not in target_levels:
                    target_levels.append(highest_level)
            return target_levels
        return [level for level in range(len(wsi_obj.level_downsamples))]

    def get_level_data(self, wsi_obj: OpenSlide, level: int) -> np.ndarray:
        """Returns data from a given level of the wsi pyramid. The data is returned in HWC format as expected by the
        tiffwriter. The data must be in HWC format otherwise the compression will not work properly.

        :param wsi_obj: The wsi object in openslide format
        :param level: The level at which the data is extracted from the wsi pyramid
        :return: A numpy array of shape (H, W, C) where H is the height, W is the width and C is the number of channels
        """
        level_data, _ = self.wsi_reader.get_data(wsi_obj, level=level)
        level_data = level_data.transpose(1, 2, 0)
        self.validate_level_data(level_data)
        return level_data

    def validate_level_data(self, level_data: np.ndarray) -> None:
        """Makes sure that the level data is in HWC format. If the level data is not in HWC format, an error is raised.
        This sanity check is necessary to avoid compression issues.
        """
        if level_data.shape[2] != 3:
            raise ValueError(
                f"Expected 3 channels but got {level_data.shape[2]}. Maybe the image is in channel first format? Try "
                "transposing the image."
            )

    def get_tiffwriter_options(self, wsi_obj: OpenSlide) -> Dict[str, Any]:
        """Returns the options that will be passed to the tiffwriter. The options are extracted from the wsi properties
        and will be written as tags in the tiff file.

        :param wsi_obj: The wsi object in openslide format
        :raises ValueError: Raises an error if the resolution unit is not in centimeters
        :return: A dictionary of options that will be passed to the tiffwriter
        """
        resolution_unit = wsi_obj.properties[self.RESOLUTION_UNIT_KEY]

        if resolution_unit != self.RESOLUTION_UNIT:
            raise ValueError(f"Resolution unit is not in {self.RESOLUTION_UNIT}: {resolution_unit}")

        options = dict(
            software=self.SOFTWARE,
            metadata={'axes': 'YXC'},
            photometric=PHOTOMETRIC.RGB,
            resolutionunit=resolution_unit,
            compression=self.compression,
            tile=(self.tile_size, self.tile_size),
        )
        return options

    def get_px_per_cm_resolution_at_level(self, wsi_obj: OpenSlide, level: int) -> Tuple[float, float]:
        """Returns the resolution of the wsi at a given level in pixels per centimeter.

        :param wsi_obj: The wsi object in openslide format
        :param level: The level at which the resolution is calculated
        :return: A tuple of floats (x_resolution, y_resolution)
        """
        um_per_cm = 10000
        um_per_px = self.wsi_reader.get_mpp(wsi_obj, level=level)
        px_per_cm = (um_per_cm / um_per_px[0], um_per_cm / um_per_px[1])
        return px_per_cm

    def convert_wsi(self, src_path: Path, tiff_path: Path) -> None:
        """Converts a single wsi file from a src format to tiff format. The tiff file is saved in the tiff_path. If the
        original src file does not have the target resolution, we skip the wsi and return None.

        :param src_path: The path to the src wsi file
        :param tiff_path: The path to the tiff file
        """

        wsi_obj = self.wsi_reader.read(src_path)

        try:
            levels = self.get_target_levels(wsi_obj)
        except ValueError as e:
            logging.warning(f"Skipping {src_path} because {e}")
            return

        options = self.get_tiffwriter_options(wsi_obj)

        with TiffWriter(tiff_path, bigtiff=True) as tif:
            for i, level in enumerate(levels):
                level_data = self.get_level_data(wsi_obj, level)
                self.validate_level_data(level_data)
                resolution = self.get_px_per_cm_resolution_at_level(wsi_obj, level)
                # the subfiletype parameter is a bitfield that determines if the wsi_level is a reduced version of
                # another image. level 0 (i.e. i=0) is the full resolution image in the pyramid.
                tif.write(level_data, resolution=resolution, subfiletype=int(i > 0), **options)

    def __call__(self, data: Dict) -> Dict:
        src_path = Path(data[self.image_key])
        tiff_path = self.get_tiff_path(src_path)
        # if the tiff file does not exist or if it exists but is empty, we convert the wsi to tiff
        if not tiff_path.exists() or (tiff_path.exists() and tiff_path.stat().st_size <= self.min_file_size):
            self.convert_wsi(src_path, tiff_path)
        if self.verbose:
            logging.info(f"Converted {src_path} to {tiff_path}")
            logging.info(f"Source file size {src_path.stat().st_size / 1e6:.2f} MB")
            logging.info(f"Tiff file size {tiff_path.stat().st_size / 1e6:.2f} MB")
        return data
