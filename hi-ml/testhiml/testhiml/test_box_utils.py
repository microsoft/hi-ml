#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import pytest

from health_ml.utils.box_utils import Box, get_bounding_box


def test_no_zeros() -> None:
    length_x = 3
    length_y = 4
    # If no elements are zero, the bounding box will have the same shape as the original
    mask = np.random.randint(1, 10, size=(length_y, length_x))
    bbox = get_bounding_box(mask)
    assert isinstance(bbox, Box)
    expected = Box(x=0, y=0, w=length_x, h=length_y)
    assert bbox == expected


def test_bounding_box_3d() -> None:
    # passing a 3D array should cause an error to be raised
    mask_3d = np.random.randint(0, 10, size=(1, 2, 3))
    with pytest.raises(TypeError):
        get_bounding_box(mask_3d)


def test_identity_matrix() -> None:
    # passing an identity matrix will return a bounding box with the same shape as the original,
    # and xmin and ymin will both be zero
    length = 5
    mask_eye = np.eye(length)
    bbox = get_bounding_box(mask_eye)
    expected = Box(x=0, y=0, w=length, h=length)
    assert bbox == expected


def test_all_zeros() -> None:
    mask = np.zeros((2, 3))
    with pytest.raises(RuntimeError):
        get_bounding_box(mask)


def test_small_rectangle() -> None:
    mask = np.zeros((5, 5), int)
    row = 0
    height = 1
    width = 2
    col = 3
    mask[row:row + height, col:col + width] = 1
    # array([[0, 0, 0, 1, 1],
    #        [0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0]])
    bbox = get_bounding_box(mask)
    expected = Box(x=col, y=row, w=width, h=height)
    assert bbox == expected


def test_tiny_mask() -> None:
    mask = np.array(1).reshape(1, 1)
    bbox = get_bounding_box(mask)
    assert bbox.x == bbox.y == 0
    assert bbox.w == bbox.h == 1


def test_tiny_box() -> None:
    mask = np.array((
        (0, 0),
        (0, 1),
    ))
    bbox = get_bounding_box(mask)
    assert bbox.x == bbox.y == bbox.w == bbox.h == 1


def test_multiple_components() -> None:
    length = 3
    mask = np.zeros((length, length), int)
    mask[0, 0] = 1
    mask[length - 1, length - 1] = 1
    bbox = get_bounding_box(mask)
    expected = Box(x=0, y=0, w=length, h=length)
    assert bbox == expected
