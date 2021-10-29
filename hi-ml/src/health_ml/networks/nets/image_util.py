#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from health.common import common_util
from health.common.common_util import any_pairwise_larger
from health.common.type_annotations import TupleFloat3, TupleFloat9, TupleInt2, TupleInt3
from health.config import PaddingMode


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


def pad_images_for_inference(images: np.ndarray,
                             crop_size: TupleInt3,
                             output_size: Optional[TupleInt3],
                             padding_mode: PaddingMode = PaddingMode.Zero) -> np.ndarray:
    """
    Pad the original image to ensure that the size of the model output as the original image.
    Padding is needed to allow the patches on the corners of the image to be handled correctly, as the model response
    for each patch will only cover the center of  the input voxels for that patch. Hence, add a padding of size
    ceil(output_size - crop_size / 2) around the original image is needed to ensure that the output size of the model
    is the same as the original image size.

    :param images: the image(s) to be padded, in shape: Z x Y x X or batched in shape: Batches x Z x Y x X.
    :param crop_size: the shape of the patches that will be taken from this image.
    :param output_size: the shape of the response for each patch from the model.
    :param padding_mode: a valid numpy padding mode.
    :return: padded copy of the original image.
    """

    def create_padding_vector() -> Tuple[TupleInt2, TupleInt2, TupleInt2]:
        """
        Creates the padding vector.
        """
        diff = np.subtract(crop_size, output_size)  # notype
        pad: List[int] = np.ceil(diff / 2.0).astype(int)
        return (pad[0], diff[0] - pad[0]), (pad[1], diff[1] - pad[1]), (pad[2], diff[2] - pad[2])

    if images is None:
        raise Exception("Image must not be none")

    if output_size is None:
        raise Exception("Output size must not be none")

    if not len(images.shape) in [3, 4]:
        raise Exception("Image must be either 3 dimensions (Z x Y x X) or "
                        "Batched into 4 dimensions (Batches x Z x Y x X)")

    if any_pairwise_larger(output_size, crop_size):
        raise Exception("crop_size must be >= output_size, found crop_size:{}, output_size:{}"
                        .format(crop_size, output_size))

    return _pad_images(images=images, padding_vector=create_padding_vector(), padding_mode=padding_mode)


def pad_images(images: np.ndarray,
               output_size: Optional[TupleInt3],
               padding_mode: PaddingMode = PaddingMode.Zero) -> np.ndarray:
    """
    Pad the original images such that their shape after padding is equal to a fixed `output_size`,
    using the provided padding mode.

    :param images: the image(s) to be padded, in shape: Z x Y x X or batched in shape: Batches x Z x Y x X.
    :param output_size: the target output shape after padding.
    :param padding_mode: a valid numpy padding mode
    :return: padded copy of the original image.
    """

    def create_padding_vector() -> Tuple[TupleInt2, TupleInt2, TupleInt2]:
        """
        Creates the padding vector ceil(crop_size - output_size / 2)
        """
        image_spatial_shape = images.shape[-3:]
        diff = np.clip(np.subtract(output_size, image_spatial_shape), a_min=0, a_max=None)
        pad: List[int] = np.ceil(diff / 2.0).astype(int)
        return (pad[0], diff[0] - pad[0]), (pad[1], diff[1] - pad[1]), (pad[2], diff[2] - pad[2])

    if images is None:
        raise Exception("Image must not be none")

    if output_size is None:
        raise Exception("Output size must not be none")

    if not len(images.shape) in [3, 4]:
        raise Exception("Image must be either 3 dimensions (Z x Y x X) or "
                        "Batched into 4 dimensions (Batches x Z x Y x X)")

    return _pad_images(images=images, padding_vector=create_padding_vector(), padding_mode=padding_mode)


def _pad_images(images: np.ndarray,
                padding_vector: Tuple[TupleInt2, TupleInt2, TupleInt2],
                padding_mode: PaddingMode) -> np.ndarray:
    """
    Pad the original images w.r.t to the padding_vector provided for padding on each side in each dimension.

    :param images: the image(s) to be padded, in shape: Z x Y x X or batched in shape: Batches x Z x Y x X.
    :param padding_vector: padding before and after in each dimension eg: ((2,2), (3,3), (2,0))
    will pad 4 pixels in Z (2 on each side), 6 pixels in Y (3 on each side)
    and 2 in X (2 on the left and 0 on the right).
    :param padding_mode: a valid numpy padding mode.
    :return: padded copy of the original image.
    """
    def pad_fn(_images: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.pad(array=x, pad_width=padding_vector, mode=padding_mode.value) for x in _images])

    # add a batch dimension if required
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        images = pad_fn(images)
        images = np.squeeze(images, axis=0)
    else:
        images = pad_fn(images)

    return images
