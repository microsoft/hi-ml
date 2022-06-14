#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pydicom as dicom
from PIL import Image
from SimpleITK import GetArrayFromImage, ReadImage
from skimage import io


def remap_to_uint8(array: np.ndarray, percentiles: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Remap values in input so the output range is ``[0, 255]``.

    :param: array: Input array.
    :param: percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = (
                'The value for percentiles should be a sequence of length 2,'
                f' but has length {len_percentiles}'
            )
            raise ValueError(message)
        a, b = percentiles
        if a >= b:
            raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
        if a < 0 or b > 100:
            raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
        cutoff: np.ndarray = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    array /= array.max()
    array *= 255
    return array.astype(np.uint8)


def load_image(img_path: str) -> Image.Image:
    """
    Load an image.

    :param: img_path: path to image
    :returns: image as PIL Image
    """
    if Path(img_path).suffix in [".jpg", ".png"]:
        image = io.imread(img_path)
    elif Path(img_path).suffixes == [".nii", ".gz"]:
        image = GetArrayFromImage(ReadImage(str(img_path)))
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
        assert image.ndim == 2
    elif Path(img_path).suffix == ".dcm":
        image = dicom.dcmread(img_path).pixel_array
    else:
        raise ValueError(f"Image type not supported, filename was: {img_path}")

    image = remap_to_uint8(image)
    return Image.fromarray(image).convert("L")
