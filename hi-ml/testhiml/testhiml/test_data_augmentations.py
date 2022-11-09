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
                            seed: int = 0) -> None:
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
    assert torch.allclose(augmented_img, expected_output_img, atol=1e-04)

    # After applying a stochastic augmentation a second time it should have a different output
    if stochastic:
        augmented_img = data_augmentation(input_img)  # type: ignore
        assert not torch.allclose(augmented_img, expected_output_img, atol=1e-04)


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
    expected_output_img1 = torch.Tensor(
        [[[[0.9639, 0.4130],
          [0.9134, 1.0000]],
         [[0.3125, 0.0000],
          [0.4474, 0.1820]],
         [[0.9195, 0.5265],
          [0.9118, 0.8291]]]])
    expected_output_img2 = torch.Tensor(
        [[[[0.8411, 0.2361],
          [0.7857, 0.8766]],
         [[0.7075, 0.0000],
          [1.0000, 0.4138]],
         [[0.9694, 0.4674],
          [0.9577, 0.8476]]]])
    expected_output_bag = torch.vstack([expected_output_img1,
                                        expected_output_img2])

    _test_data_augmentation(data_augmentation,
                            dummy_bag,
                            expected_output_bag,
                            stochastic=True)


def test_gaussian_blur() -> None:
    data_augmentation = GaussianBlur(3, p=1.0)
    expected_output_img = torch.Tensor(
        [[[[0.8302, 0.7639],
          [0.8149, 0.6943]],
         [[0.7423, 0.6225],
          [0.6815, 0.6094]],
         [[0.7821, 0.6929],
          [0.7393, 0.7463]]]])

    _test_data_augmentation(data_augmentation, dummy_img, expected_output_img, stochastic=True)


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
