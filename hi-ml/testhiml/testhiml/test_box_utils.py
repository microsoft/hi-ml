#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import pytest

from health_ml.utils.box_utils import Box, get_bounding_box


def test_get_bounding_box() -> None:
    length_x = 3
    length_y = 4
    # If no elements are zero, the bounding box will have the same shape as the original
    mask = np.random.randint(1, 10, size=(length_x, length_y))
    bbox = get_bounding_box(mask)
    assert isinstance(bbox, Box)
    assert bbox.w == length_x
    assert bbox.h == length_y

    # passing a 3D array should cause an error to be raised
    length_z = 5
    mask_3d = np.random.randint(0, 10, size=(length_x, length_y, length_z))
    with pytest.raises(TypeError):
        get_bounding_box(mask_3d)

    # passing an identity matrix will return a bounding box with the same shape as the original,
    # and xmin and ymin will both be zero
    mask_eye = np.eye(length_x)
    bbox_eye = get_bounding_box(mask_eye)
    assert isinstance(bbox_eye, Box)
    assert bbox_eye.w == length_x
    assert bbox_eye.h == length_x
    assert bbox_eye.x == bbox_eye.y == 0
