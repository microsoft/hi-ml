import torch

from health_multimodal.image.model.pretrained import get_imagenet_init_encoder
from health_multimodal.image.model.resnet import resnet50


def test_get_imagenet_init_encoder() -> None:
    """Test that the ``imagenet`` option loads weights correctly."""
    expected_model = resnet50(pretrained=True)
    imagenet_model = get_imagenet_init_encoder()

    for imagenet_param, expected_param in zip(
        imagenet_model.encoder.encoder.named_parameters(), expected_model.named_parameters()
    ):
        assert imagenet_param[0] == expected_param[0]
        assert torch.isclose(imagenet_param[1], expected_param[1]).all()
