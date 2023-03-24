#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pydicom as dicom
from PIL import Image
import SimpleITK as sitk
from skimage import io


def remap_to_uint8(array: np.ndarray, percentiles: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Remap values in input so the output range is :math:`[0, 255]`.

    Percentiles can be used to specify the range of values to remap.
    This is useful to discard outliers in the input data.

    :param array: Input array.
    :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = 'The value for percentiles should be a sequence of length 2,' f' but has length {len_percentiles}'
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


def load_image(path: Path) -> Image.Image:
    """Load an image from disk.

    The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

    :param path: Path to image.
    :returns: Image as ``Pillow`` ``Image``.
    """
    # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
    if path.suffix in [".jpg", ".jpeg", ".png"]:
        image = io.imread(path)
    elif path.suffixes == [".nii", ".gz"]:
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
        assert image.ndim == 2
    elif path.suffix == ".dcm":
        image = dicom.dcmread(path).pixel_array
    else:
        raise ValueError(f"Image type not supported, filename was: {path}")

    image = remap_to_uint8(image)
    return Image.fromarray(image).convert("L")
