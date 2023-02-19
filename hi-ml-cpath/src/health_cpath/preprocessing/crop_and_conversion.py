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
from typing import Any, Dict, List, Optional


AMPERSAND = "&"
UNDERSCORE = "_"


class WSIFormat(str, Enum):
    """The format of the wsi file."""
    NDPI = "ndpi"
    SVS = "svs"
    TIF = "tif"
    TIFF = "tiff"


class ConvertToTiffd(MapTransform):
    """Converts a wsi file to a tiff file at a given magnification level. Additionally, the highest resolution level of
    the wsi is also saved in the tiff file if the highest resolution level is not equal to the target resolution and
    add_lowest_magnification is True. The tiff files are saved in the dest_dir.
    Openslide backend is used to read the original wsi files.
    """
    def __init__(
        self,
        dest_dir: Path,
        image_key: str = SlideKey.IMAGE,
        src_format: WSIFormat = WSIFormat.NDPI,
        target_magnifications: Optional[List[float]] = [10.],
        add_lowest_magnification: bool = False,
        replace_ampersand_by: str = AMPERSAND,
    ) -> None:
        self.dest_dir = dest_dir
        self.src_format = src_format
        self.image_key = image_key
        self.target_magnifications = target_magnifications
        self.wsi_reader = WSIReader(WSIBackend.OPENSLIDE)
        self.add_lowest_magnification = add_lowest_magnification
        self.replace_ampersand_by = replace_ampersand_by

    def get_tiff_path(self, src_path: Path) -> Path:
        tiff_filename = src_path.name.replace(self.src_format, WSIFormat.TIFF.value)
        tiff_filename = tiff_filename.replace(AMPERSAND, self.replace_ampersand_by)
        return self.dest_dir / tiff_filename

    def _get_target_level(self, wsi_obj: OpenSlide, target_magnification: float) -> int:
        """Returns the level of the wsi pyramid that is equal to the target magnification.

        :param wsi_obj: the wsi object in openslide format
        :param target_magnification: the target magnification e.g. 10x
        """
        base_objective_power = int(wsi_obj.properties['openslide.objective-power'])
        # TODO add objective-power to the properties of the wsi object in monai
        for level_idx, level_downsample in enumerate(wsi_obj.level_downsamples):
            objective_power = base_objective_power / level_downsample
            if math.isclose(objective_power, target_magnification, rel_tol=1e-3):
                return level_idx
        raise ValueError(f"Target magnification {target_magnification} not found")

    def _get_highest_level(self, wsi_obj: OpenSlide) -> int:
        """Returns the highest resolution level of the wsi pyramid."""
        return len(wsi_obj.level_downsamples) - 1

    def _get_level_data(self, wsi_obj: OpenSlide, level: int) -> np.ndarray:
        """Returns data from a given level of the wsi pyramid. The data is returned in HWC format as expected by the
        tiffwriter. The data must be in HWC format otherwise the compression will not work properly.

        :param wsi_obj: The wsi object in openslide format
        :param level: The level at which the data is extracted from the wsi pyramid
        :return: A numpy array of shape (H, W, C) where H is the height, W is the width and C is the number of channels
        """
        level_data, _ = self.wsi_reader.get_data(wsi_obj, level=level)
        return level_data.transpose(1, 2, 0)

    def _get_target_levels(self, wsi_obj: Any) -> List[int]:
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

    def convert_wsi(self, src_path: Path, tiff_path: Path) -> None:
        """Converts a single wsi file from a src format to tiff format. The tiff file is saved in the tiff_path. If the
        original src file does not have the target resolution, we skip the wsi and return None.

        :param src_path: The path to the src wsi file
        :param tiff_path: The path to the tiff file
        """

        wsi_obj = self.wsi_reader.read(src_path)
        try:
            levels = self._get_target_levels(wsi_obj)
        except ValueError as e:
            logging.warning(f"Skipping {src_path} because {e}")
            return

        resolution_unit = wsi_obj.properties['#.ResolutionUnit']
        assert resolution_unit == 'centimeter', f"Resolution unit is not in centimeters: {resolution_unit}"
        um_per_cm = 10000.
        options = dict(
            software='tifffile',
            metadata={'axes': 'YXC'},
            photometric=PHOTOMETRIC.RGB,
            resolutionunit=resolution_unit,
            compression=COMPRESSION.ADOBE_DEFLATE,  # ADOBE_DEFLATE aka ZLIB lossless compression
            tile=(512, 512),  # 512x512 tiles
        )
        with TiffWriter(tiff_path, bigtiff=True) as tif:
            for i, level in enumerate(levels):
                level_data = self._get_level_data(wsi_obj, level)
                assert level_data.shape[2] == 3, f"Expected 3 channels but got {level_data.shape[2]}. Maybe the image" \
                                                 " is in channel first format? Try transposing the image."
                um_per_px = self.wsi_reader.get_mpp(wsi_obj, level=level)
                px_per_cm = (um_per_cm / um_per_px[0], um_per_cm / um_per_px[1])
                # the subfiletype parameter is a bitfield that determines if the wsi_level is a reduced version of
                # another image. level 0 (i.e. i=0) is the full resolution image in the pyramid.
                tif.write(level_data, resolution=px_per_cm, subfiletype=int(i > 0), **options)

    def __call__(self, data: Dict) -> Dict:
        src_path = Path(data[self.image_key])
        tiff_path = self.get_tiff_path(src_path)
        if not tiff_path.exists():
            self.convert_wsi(src_path, tiff_path)
        return data
