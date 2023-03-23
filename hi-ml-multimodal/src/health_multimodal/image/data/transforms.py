#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Callable, Sequence, Optional, Tuple

import torch
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop


class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)


def create_chest_xray_transform_for_inference(resize: int, center_crop_size: int) -> Compose:
    """
    Defines the image transformation pipeline for Chest-Xray datasets.

    :param resize: The size to resize the image to. Linear resampling is used.
                   Resizing is applied on the axis with smaller shape.
    :param center_crop_size: The size to center crop the image to. Square crop is applied.
    """

    transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels()]
    return Compose(transforms)


def infer_resize_params(val_img_transforms: Sequence[Callable]) -> Tuple[Optional[int], Optional[int]]:
    """
    Given the validation transforms pipeline, extract the sizes to which the image was resized and cropped, if any.
    """
    resize_size_from_transforms = None
    crop_size_from_transforms = None
    supported_types = Resize, CenterCrop, ToTensor, ExpandChannels
    for transform in val_img_transforms:
        trsf_type = type(transform)
        if trsf_type not in supported_types:
            raise ValueError(f"Unsupported transform type {trsf_type}. Supported types are {supported_types}")
        if isinstance(transform, Resize):
            if resize_size_from_transforms is None and crop_size_from_transforms is None:
                assert transform.max_size is None
                assert isinstance(transform.size, int), f"Expected int, got {transform.size}"
                resize_size_from_transforms = transform.size
            else:
                raise ValueError("Expected Resize to be the first transform if present in val_img_transforms")
        elif isinstance(transform, CenterCrop):
            if crop_size_from_transforms is None:
                two_dims = len(transform.size) == 2
                same_sizes = transform.size[0] == transform.size[1]
                is_square = two_dims and same_sizes
                assert is_square, "Only square center crop supported"
                crop_size_from_transforms = transform.size[0]
            else:
                raise ValueError(
                    f"Crop size has already been set to {crop_size_from_transforms} in a previous transform"
                )

    return resize_size_from_transforms, crop_size_from_transforms
