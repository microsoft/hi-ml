from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import SimpleITK as sitk
from PIL import Image

from health_multimodal.image.data.io import load_image, remap_to_uint8


def _assert_min_max_dtype(array: np.ndarray) -> None:
    assert array.min() == 0
    assert array.max() == 255
    assert array.dtype == np.uint8


def test_load_image() -> None:
    """
    Tests the image loading function using dummy NIFTI and JPG files.
    """

    def _assertions(path: str) -> None:
        img = load_image(path)
        assert img.size == size
        array = np.asarray(img)
        _assert_min_max_dtype(array)

    size = 4, 4
    array = np.arange(16, dtype=np.uint8).reshape(*size)
    image = Image.fromarray(array).convert('RGB')
    with NamedTemporaryFile(suffix='.jpg') as file:
        image.save(file)
        _assertions(file.name)

    nifti_img = sitk.GetImageFromArray(np.arange(16, dtype=np.uint16).reshape(*size) + 100)
    with NamedTemporaryFile(suffix='.nii.gz') as file:
        sitk.WriteImage(nifti_img, file.name)
        _assertions(file.name)


def test_remap_to_uint8() -> None:
    """
    Tests the intensity casting function using different percentiles.
    """
    array = np.arange(10).astype(np.uint16)  # mimic DICOM data type
    with pytest.raises(ValueError):
        remap_to_uint8(array, (1, 2, 3))
    with pytest.raises(ValueError):
        remap_to_uint8(array, (-1, 50))
    with pytest.raises(ValueError):
        remap_to_uint8(array, (1, 150))
    with pytest.raises(ValueError):
        remap_to_uint8(array, (5, 2))
    normalized = remap_to_uint8(array)
    _assert_min_max_dtype(normalized)
    normalized = remap_to_uint8(array, (1, 99))
    _assert_min_max_dtype(normalized)

    array_positive_min = array + 5
    normalized = remap_to_uint8(array_positive_min)
    _assert_min_max_dtype(normalized)
    normalized = remap_to_uint8(array_positive_min, (1, 99))
    _assert_min_max_dtype(normalized)
