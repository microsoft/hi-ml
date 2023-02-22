import torch
import numpy as np
import random

from torch import Tensor
from typing import Callable

from health_ml.utils.data_augmentations import HEDJitter, StainNormalization, GaussianBlur, \
    RandomRotationByMultiplesOf90

# global dummy image
dummy_img = torch.Tensor(
    [[[[0.4767, 0.0415],
       [0.8325, 0.8420]],
      [[0.9859, 0.9119],
       [0.8717, 0.9098]],
      [[0.1592, 0.7216],
       [0.8305, 0.1127]]]])
dummy_bag = torch.stack([dummy_img.squeeze(0), dummy_img.squeeze(0)])


def _test_data_augmentation(data_augmentation: Callable[[Tensor], Tensor],
                            input_img: torch.Tensor,
                            expected_output_img: torch.Tensor,
                            stochastic: bool,
                            seed: int = 0,
                            atol: float = 1e-04) -> None:
    if stochastic:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    augmented_img = data_augmentation(input_img)

    # Check types
    assert torch.is_tensor(augmented_img)
    assert input_img.dtype == augmented_img.dtype

    # Check shape
    assert input_img.shape == augmented_img.shape

    # Check range
    assert augmented_img.max() <= 1.0
    assert augmented_img.min() >= 0.0

    # Check if the transformation still produces the same output
    assert torch.allclose(augmented_img, expected_output_img, atol=atol)

    # After applying a stochastic augmentation a second time it should have a different output
    if stochastic:
        augmented_img = data_augmentation(input_img)  # type: ignore
        assert not torch.allclose(augmented_img, expected_output_img, atol=atol)


def test_stain_normalization() -> None:
    data_augmentation = StainNormalization()
    expected_output_img = torch.Tensor(
        [[[[0.8627, 0.4510],
          [0.8314, 0.9373]],
         [[0.6157, 0.2353],
          [0.8706, 0.4863]],
         [[0.8235, 0.5294],
          [0.8275, 0.7725]]]])
    expected_output_bag = torch.stack([expected_output_img.squeeze(0), expected_output_img.squeeze(0)])

    _test_data_augmentation(data_augmentation, dummy_img, expected_output_img, stochastic=False)
    _test_data_augmentation(data_augmentation, dummy_bag, expected_output_bag, stochastic=False)

    # Test tiling on the fly (i.e. when the input image does not have a batch dimension)
    _test_data_augmentation(data_augmentation, dummy_img.squeeze(0), expected_output_img.squeeze(0), stochastic=False)


def test_hed_jitter() -> None:
    data_augmentation = HEDJitter(0.05)
    expected_output_img = torch.Tensor(
        [[[[0.3375, 0.0000],
          [0.6677, 0.3245]],
         [[0.2473, 0.0031],
          [1.0000, 0.1836]],
         [[0.0877, 0.1219],
          [0.6196, 0.0418]]]])

    _test_data_augmentation(data_augmentation, dummy_img, expected_output_img, stochastic=True, seed=1, atol=1e-3)

    # Test tiling on the fly (i.e. when the input image does not have a batch dimension)
    _test_data_augmentation(data_augmentation, dummy_img.squeeze(0), expected_output_img.squeeze(0),
                            stochastic=True, seed=1, atol=1e-3)


def test_gaussian_blur() -> None:
    data_augmentation = GaussianBlur(3, p=1.0)
    expected_output_img = torch.Tensor(
        [[[[0.5953, 0.6321],
          [0.4528, 0.5125]],
         [[0.9126, 0.9138],
          [0.9229, 0.9299]],
         [[0.4343, 0.4848],
          [0.4601, 0.4448]]]])

    _test_data_augmentation(data_augmentation, dummy_img, expected_output_img, stochastic=True, seed=1, atol=1e-3)

    # Test tiling on the fly (i.e. when the input image does not have a batch dimension)
    _test_data_augmentation(data_augmentation, dummy_img.squeeze(0), expected_output_img.squeeze(0),
                            stochastic=True, seed=1, atol=1e-3)


def test_random_rotation() -> None:
    data_augmentation = RandomRotationByMultiplesOf90()
    expected_output_img = torch.Tensor(
        [[[[0.0415, 0.8420],
          [0.4767, 0.8325]],
         [[0.9119, 0.9098],
          [0.9859, 0.8717]],
         [[0.7216, 0.1127],
          [0.1592, 0.8305]]]])

    _test_data_augmentation(data_augmentation, dummy_img, expected_output_img, stochastic=True, seed=1)
