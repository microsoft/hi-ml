#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
import SimpleITK as sitk
from SSL.datamodules_and_datasets.io_util import PhotometricInterpretation
from health_azure.utils import PathOrString


def tests_root_directory(path: Optional[PathOrString] = None) -> Path:
    """
    Gets the full path to the root directory that holds the tests.
    If a relative path is provided then concatenate it with the absolute path
    to the repository root.

    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    root = Path(os.path.realpath(__file__)).parent.parent
    return root / path if path else root


def write_test_dicom(array: np.ndarray, path: Path, is_monochrome2: bool = True,
                     bits_stored: Optional[int] = None) -> None:
    """
    This saves the input array as a Dicom file.
    This function DOES NOT create a usable Dicom file and is meant only for testing: tags are set to
    random/default values so that pydicom does not complain when reading the file.
    """

    # Write a file directly with pydicom is cumbersome (all tags need to be set by hand). Hence using simpleITK to
    # create the file. However SimpleITK does not let you set the tags directly, so using pydicom so set them after.
    image = sitk.GetImageFromArray(array)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path))
    writer.Execute(image)

    ds = pydicom.dcmread(path)
    ds.PhotometricInterpretation = PhotometricInterpretation.MONOCHROME2.value if is_monochrome2 else \
        PhotometricInterpretation.MONOCHROME1.value
    if bits_stored is not None:
        ds.BitsStored = bits_stored
    ds.save_as(path)
