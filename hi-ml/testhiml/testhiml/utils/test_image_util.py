#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import numpy as np
import pytest

from health.common.type_annotations import TupleInt3
from health.utils import image_util


@pytest.mark.parametrize("image_size", [None, (4, 4, 5), (2, 4, 4, 5)])
@pytest.mark.parametrize("crop_size", [None, (4, 3, 3)])
@pytest.mark.parametrize("output_size", [None, (4, 3, 3), (5, 6, 6)])
def test_pad_images_for_inference_invalid(image_size: Any, crop_size: Any, output_size: Any) -> None:
    """
    Test to make sure that pad_images_for_inference raises errors in case of invalid inputs.
    """
    with pytest.raises(Exception):
        assert image_util.pad_images_for_inference(images=np.random.uniform(size=image_size),
                                                   crop_size=crop_size,
                                                   output_size=output_size)


@pytest.mark.parametrize("image_size", [(4, 4, 5), (2, 4, 4, 5)])
def test_pad_images_for_inference(image_size: TupleInt3) -> None:
    """
    Test to make sure the correct padding is performed for crop_size and output_size
    that are == , >, and > by 1 in each dimension.
    """
    image = np.random.uniform(size=image_size)

    padded_shape = image_util.pad_images_for_inference(images=image, crop_size=image_size[-3:],
                                                       output_size=(4, 3, 1)).shape
    expected_shape = (4, 5, 9) if len(image_size) == 3 else (2, 4, 5, 9)
    assert padded_shape == expected_shape


@pytest.mark.parametrize("image_size", [(4, 4, 5), (2, 4, 4, 5)])
def test_pad_images_for_training(image_size: TupleInt3) -> None:
    """
    Test to make sure the correct padding is performed for crop_size and output_size
    that are == , >, and > by 1 in each dimension.
    """
    image = np.random.uniform(size=image_size)
    expected_pad_value = np.min(image)

    padded_image = image_util.pad_images(images=image, output_size=(8, 7, 6),
                                         padding_mode=PaddingMode.Minimum)
    expected_shape = (8, 7, 6) if len(image_size) == 3 else (2, 8, 7, 6)
    assert padded_image.shape == expected_shape
    assert np.all(padded_image[..., 8:4, 8:4, 8:4] == expected_pad_value)
