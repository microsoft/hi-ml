#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum
from typing import Iterable

import numpy as np
import pydicom as dicom

from health_azure.utils import PathOrString


class PhotometricInterpretation(Enum):
    MONOCHROME1 = "MONOCHROME1"
    MONOCHROME2 = "MONOCHROME2"


class DicomFileType(Enum):
    """
    Supported file extensions that indicate Dicom data.
    """
    Dicom = ".dcm"


VALID_DICOM_EXTENSIONS_TUPLE = tuple([f.value for f in DicomFileType])


def _file_matches_extension(file: PathOrString, valid_extensions: Iterable[str]) -> bool:
    """
    Returns true if the given file name has any of the provided file extensions.

    :param file: The file name to check.
    :param valid_extensions: A tuple with all the extensions that are considered valid.
    :return: True if the file has any of the given extensions.
    """
    dot = "."
    extensions_with_dot = tuple(e if e.startswith(dot) else dot + e for e in valid_extensions)
    return str(file).lower().endswith(extensions_with_dot)


def is_dicom_file_path(file: PathOrString) -> bool:
    """
    Returns true if the given file name appears to belong to a Dicom file.
    This is done based on extensions only. The file does not need to exist.

    :param file: The file name to check.
    :return: True if the file name indicates a Dicom file.
    """
    return _file_matches_extension(file, VALID_DICOM_EXTENSIONS_TUPLE)


def load_dicom_image(path: PathOrString) -> np.ndarray:
    """
    Loads an array from a single dicom file.
    :param path: The path to the dicom file.
    """
    ds = dicom.dcmread(path)
    pixels = ds.pixel_array
    bits_stored = int(ds.BitsStored)  # type: ignore
    if ds.PhotometricInterpretation == PhotometricInterpretation.MONOCHROME1.value:
        pixel_repr = ds.PixelRepresentation
        if pixel_repr == 0:  # unsigned
            pixels = 2 ** bits_stored - 1 - pixels
        elif pixel_repr == 1:  # signed
            pixels = -1 * (pixels + 1)
        else:
            raise ValueError("Unknown value for DICOM tag 0028,0103 PixelRepresentation")
    # Return a float array, we may resize this in load_3d_images_and_stack, and interpolation will not work on int
    return pixels.astype(np.float)  # type: ignore
