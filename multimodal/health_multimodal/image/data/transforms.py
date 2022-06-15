from typing import Callable

import torch
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.transforms import CenterCrop


class ExpandChannels:
    """
    Transforms an image with 1 channel to an image with 3 channels by copying pixel intensities of the image along
    the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param: data of shape [1, H, W]
        :return: data with channel copied 3 times, shape [3, H, W]
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)


def create_chest_xray_transform_for_inference(resize: int, center_crop_size: int) -> Callable:
    """
    Defines the image transformation pipeline for Chest-Xray datasets.
    :param resize: The size to resize the image to. Linear resampling is used.
                   Resizing is applied on the axis with smaller shape.
    :param center_crop_size: The size to center crop the image to. Square crop is applied.
    """

    transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels()]
    return Compose(transforms)
