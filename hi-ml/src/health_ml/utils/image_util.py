#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import torch

from health_ml.common import common_util
from health_ml.common.type_annotations import TupleFloat3, TupleFloat9


NumpyOrTorch = Union[np.ndarray, torch.Tensor]
Range = Tuple[Union[int, float], Union[int, float]]

# Factor by which array range bounds can be exceeded without triggering an error. If the range is [low, high], we
# only raise an exception if values are outside the range [low-delta, high+delta], where
# delta = (high-low) * VALUE_RANGE_TOLERANCE. Otherwise, we take max with low and min with high, to force all
# values to be inside the bounds.
VALUE_RANGE_TOLERANCE = 1e-6


@dataclass
class ImageHeader:
    """
    A 3D image header
    """
    spacing: TupleFloat3  # Z x Y x X
    origin: TupleFloat3  # X x Y x Z
    direction: TupleFloat9  # X x Y x Z

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)


def get_unit_image_header(spacing: Optional[TupleFloat3] = None) -> ImageHeader:
    """
    Creates an ImageHeader object with the origin at 0, and unit direction. The spacing is set to the argument,
    defaulting to (1, 1, 1) if not provided.
    :param spacing: The image spacing, as a (Z, Y, X) tuple.
    """
    if not spacing:
        spacing = (1, 1, 1)
    return ImageHeader(origin=(0, 0, 0), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=spacing)


class ImageDataType(Enum):
    """
    Data type for medical image data (e.g. masks and labels)
    Segmentation maps are one-hot encoded.
    """
    IMAGE = np.float32
    SEGMENTATION = np.float32
    MASK = np.uint8
    CLASSIFICATION_LABEL = np.float32
