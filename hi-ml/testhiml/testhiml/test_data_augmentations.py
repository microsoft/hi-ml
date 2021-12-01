import torch
import numpy as np
import random

from typing import Callable

from health_ml.utils.data_augmentations import HEDJitter, StainNormalization


# global dummy image
dummy_img = torch.Tensor(
        [[[[0.4767, 0.0415],
          [0.8325, 0.8420]],
         [[0.9859, 0.9119],
          [0.8717, 0.9098]],
         [[0.1592, 0.7216],
          [0.8305, 0.1127]]]])

def _test_data_augmentation(data_augmentation: Callable[[torch.Tensor], torch.Tensor],
                            input_img: torch.Tensor,
                            expected_output_img: torch.Tensor,
                            stochastic: bool) -> None:
    if stochastic:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

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
        assert not(torch.allclose(augmented_img, expected_output_img, atol=1e-04))


def test_stain_normalization() -> None:
    data_augmentation = StainNormalization()
    expected_output_img = torch.Tensor(
        [[[[0.8627, 0.4510],
          [0.8314, 0.9373]],
         [[0.6157, 0.2353],
          [0.8706, 0.4863]],
         [[0.8235, 0.5294],
          [0.8275, 0.7725]]]])

    _test_data_augmentation(data_augmentation, dummy_img, expected_output_img, stochastic=False)

def test_hed_jitter() -> None:
    data_augmentation = HEDJitter(0.05)
    expected_output_img = torch.Tensor(
        [[[[0.4536, 0.0221],
          [0.8084, 0.8164]],
         [[0.9781, 0.9108],
          [0.8522, 0.8933]],
         [[0.1138, 0.6730],
          [0.7773, 0.0666]]]])

    _test_data_augmentation(data_augmentation, dummy_img, expected_output_img, stochastic=True)
