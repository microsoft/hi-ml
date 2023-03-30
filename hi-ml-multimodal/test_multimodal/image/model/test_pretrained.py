import pytest
import torch

from health_multimodal.image.model.model import ImageModel
from health_multimodal.image.model.pretrained import get_biovil_t_linear_image_classifier


@pytest.fixture
def dummy_input_tensor() -> torch.Tensor:
    batch_size = 2
    return torch.randn(batch_size, 3, 448, 448)

@pytest.fixture
def biovil_t_linear_image_classifier() -> ImageModel:
    # Set the path to None to initialize the model weights with random values
    biovil_t_checkpoint_path = None
    return get_biovil_t_linear_image_classifier(biovil_t_checkpoint_path)

def test_get_biovil_t_linear_image_classifier_shape(biovil_t_linear_image_classifier: ImageModel,
                                                    dummy_input_tensor: torch.Tensor) -> None:
    num_classes = 2
    num_tasks = 8

    output = biovil_t_linear_image_classifier(dummy_input_tensor)
    expected_shape = torch.Size([dummy_input_tensor.shape[0], num_classes, num_tasks])

    assert output.class_logits.shape == expected_shape, \
        f"Unexpected output shape. Expected {expected_shape} but got {output.class_logits.shape}"

def test_get_biovil_t_linear_image_classifier_inference(biovil_t_linear_image_classifier: ImageModel,
                                                        dummy_input_tensor: torch.Tensor) -> None:
    with torch.no_grad():
        try:
            _ = biovil_t_linear_image_classifier(dummy_input_tensor)
        except Exception as e:
            pytest.fail(f"Model inference failed: {e}")
