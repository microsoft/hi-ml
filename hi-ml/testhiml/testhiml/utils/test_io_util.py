#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import SimpleITK as sitk
import numpy as np
import pydicom
import pytest

from health_ml.common.fixed_paths_for_tests import full_ml_test_data_path
from health_ml.common.output_directories import OutputFolderForTests
from health_ml.utils import io_util
from health_ml.utils.io_util import PhotometricInterpretation, \
    is_dicom_file_path, is_nifti_file_path, is_numpy_file_path, load_dicom_image, load_image_in_known_formats, \
    load_numpy_image, reverse_tuple_float3, load_dicom_series_and_save
from testhiml.utils.util import assert_file_contains_string


known_nii_path = full_ml_test_data_path("test_good.nii.gz")
known_array = np.ones((128, 128, 128))
bad_nii_path = full_ml_test_data_path("test_bad.nii.gz")
good_npy_path = full_ml_test_data_path("test_good.npz")
# A sample H&N DICOM series,
dicom_series_folder = full_ml_test_data_path() / "dicom_series_data" / "HN"
# A sample H&N segmentation
HNSEGMENTATION_FILE = full_ml_test_data_path() / "dicom_series_data" / "hnsegmentation.nii.gz"


@pytest.mark.parametrize("path", ["", " ", None, "not_exists", ".", "tests/test_io_util.py"])
def test_bad_path_load_image(path: str) -> None:
    with pytest.raises(ValueError):
        io_util.load_nifti_image(path)


@pytest.mark.parametrize("path", [bad_nii_path])
def test_bad_image_load_image(path: Any) -> None:
    with pytest.raises(ValueError):
        io_util.load_nifti_image(path)


def test_nii_load_image() -> None:
    image_with_header = io_util.load_nifti_image(known_nii_path)
    assert np.array_equal(image_with_header.image, known_array)


def test_nii_load_zyx(test_output_dirs: OutputFolderForTests) -> None:
    expected_shape = (44, 167, 167)
    file_path = full_ml_test_data_path("patch_sampling/scan_small.nii.gz")
    image: sitk.Image = sitk.ReadImage(str(file_path))
    assert image.GetSize() == reverse_tuple_float3(expected_shape)
    img = sitk.GetArrayFromImage(image)
    assert img.shape == expected_shape
    image_header = io_util.load_nifti_image(file_path)
    assert image_header.image.shape == expected_shape
    assert image_header.header.spacing is not None
    np.testing.assert_allclose(image_header.header.spacing, (3.0, 1.0, 1.0), rtol=0.1)


@pytest.mark.parametrize("value, expected",
                         [(["apple"], "apple"),
                          (["apple", "butter"], "apple\nbutter\n")])
def test_save_file(value: Any, expected: Any) -> None:
    file = full_ml_test_data_path("test.txt")
    io_util.save_lines_to_file(Path(file), value)
    assert_file_contains_string(file, expected)
    os.remove(str(file))


@pytest.mark.parametrize("input", [("foo.txt", False),
                                   ("foo.gz", False),
                                   ("foo.nii.gz", True),
                                   ("foo.nii", True),
                                   ("nii.gz", False),
                                   ])
def test_is_nifti_file(input: Tuple[str, bool]) -> None:
    file, expected = input
    assert is_nifti_file_path(file) == expected
    assert is_nifti_file_path(Path(file)) == expected


@pytest.mark.parametrize("input", [("foo.npy", True),
                                   ("foo.mnpy", False),
                                   ("npy", False),
                                   ("foo.txt", False),
                                   ])
def test_is_numpy_file(input: Tuple[str, bool]) -> None:
    file, expected = input
    assert is_numpy_file_path(file) == expected
    assert is_numpy_file_path(Path(file)) == expected


def test_load_numpy_image(test_output_dirs: OutputFolderForTests) -> None:
    array_size = (20, 30, 40)
    array = np.ones(array_size)
    assert array.shape == array_size
    npy_file = test_output_dirs.root_dir / "file.npy"
    assert is_numpy_file_path(npy_file)
    np.save(npy_file, array)
    image = load_numpy_image(npy_file)
    assert image.shape == array_size
    image_and_segmentation = load_image_in_known_formats(npy_file, load_segmentation=False)
    assert image_and_segmentation.images.shape == array_size


@pytest.mark.parametrize("input", [("foo.dcm", True),
                                   ("foo.mdcm", False),
                                   ("dcm", False),
                                   ("foo.txt", False),
                                   ])
def test_is_dicom_file(input: Tuple[str, bool]) -> None:
    file, expected = input
    assert is_dicom_file_path(file) == expected
    assert is_dicom_file_path(Path(file)) == expected


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


@pytest.mark.parametrize("is_signed", [True, False])
@pytest.mark.parametrize("is_monochrome2", [True, False])
def test_load_dicom_image_ones(test_output_dirs: OutputFolderForTests,
                               is_signed: bool, is_monochrome2: bool) -> None:
    """
    Test loading of 2D Dicom images filled with binary array of type (uint16) and (int16).
    """
    array_size = (20, 30)
    if not is_signed:
        array = np.ones(array_size, dtype='uint16')
        array[::2] = 0
    else:
        array = -1 * np.ones(array_size, dtype='int16')
        array[::2] = 0

    assert array.shape == array_size

    if is_monochrome2:
        to_write = array
    else:
        if not is_signed:
            to_write = np.zeros(array_size, dtype='uint16')
            to_write[::2] = 1
        else:
            to_write = np.zeros(array_size, dtype='int16')
            to_write[::2] = -1

    dcm_file = test_output_dirs.root_dir / "file.dcm"
    assert is_dicom_file_path(dcm_file)
    write_test_dicom(array=to_write, path=dcm_file, is_monochrome2=is_monochrome2, bits_stored=1)

    image = load_dicom_image(dcm_file)
    assert image.ndim == 2 and image.shape == array_size
    assert np.array_equal(image, array)

    image_and_segmentation = load_image_in_known_formats(dcm_file, load_segmentation=False)
    assert image_and_segmentation.images.ndim == 2 and image_and_segmentation.images.shape == array_size
    assert np.array_equal(image_and_segmentation.images, array)


@pytest.mark.parametrize("is_signed", [True, False])
@pytest.mark.parametrize("is_monochrome2", [True, False])
@pytest.mark.parametrize("bits_stored", [14, 16])
def test_load_dicom_image_random(test_output_dirs: OutputFolderForTests,
                                 is_signed: bool, is_monochrome2: bool, bits_stored: int) -> None:
    """
    Test loading of 2D Dicom images of type (uint16) and (int16).
    """
    array_size = (20, 30)
    if not is_signed:
        array = np.random.randint(0, 200, size=array_size, dtype='uint16')
    else:
        array = np.random.randint(-200, 200, size=array_size, dtype='int16')
    assert array.shape == array_size

    if is_monochrome2:
        to_write = array
    else:
        if not is_signed:
            to_write = 2 ** bits_stored - 1 - array
        else:
            to_write = -1 * array - 1

    dcm_file = test_output_dirs.root_dir / "file.dcm"
    assert is_dicom_file_path(dcm_file)
    write_test_dicom(array=to_write, path=dcm_file, is_monochrome2=is_monochrome2, bits_stored=bits_stored)

    image = load_dicom_image(dcm_file)
    assert image.ndim == 2 and image.shape == array_size
    assert np.array_equal(image, array)

    image_and_segmentation = load_image_in_known_formats(dcm_file, load_segmentation=False)
    assert image_and_segmentation.images.ndim == 2 and image_and_segmentation.images.shape == array_size
    assert np.array_equal(image_and_segmentation.images, array)


@pytest.mark.parametrize(["file_path", "expected_shape"],
                         [
                             ("train_and_test_data/id1_mask.nii.gz", (75, 75, 75)),
                             ("hdf5_data/patient_hdf5s/4be9beed-5861-fdd2-72c2-8dd89aadc1ef.h5", (4, 5, 7)),
])
def test_load_image(file_path: str, expected_shape: Tuple) -> None:
    full_file_path = full_ml_test_data_path() / file_path
    image_and_segmentation = load_image_in_known_formats(full_file_path, load_segmentation=False)
    assert image_and_segmentation.images.shape == expected_shape


def test_load_dicom_series(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that a DICOM series can be loaded.

    :param test_output_dirs: Test output directories.
    :return: None.
    """
    nifti_file = test_output_dirs.root_dir / "test_dicom_series.nii.gz"
    load_dicom_series_and_save(dicom_series_folder, nifti_file)
    expected_shape = (3, 512, 512)
    image_header = io_util.load_nifti_image(nifti_file)
    assert image_header.image.shape == expected_shape
    assert image_header.header.spacing is not None
    np.testing.assert_allclose(image_header.header.spacing, (2.5, 1.269531, 1.269531), rtol=0.1)
