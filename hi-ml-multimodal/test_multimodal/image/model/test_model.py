#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from dataclasses import fields

import pytest
import torch

from health_multimodal.image.model.model import ImageModel
from health_multimodal.image.model.model import ImageEncoder
from health_multimodal.image.model.model import ImageModelOutput
from health_multimodal.image.model.model import get_biovil_resnet
from health_multimodal.image.model.model import restore_training_mode
from health_multimodal.image.model.modules import MultiTaskModel
from health_multimodal.image.model.resnet import resnet50


def test_frozen_cnn_model() -> None:
    """
    Checks if the mode of module parameters is set correctly.
    """

    model = ImageModel(img_model_type='resnet18',
                       joint_feature_size=4,
                       num_classes=2,
                       freeze_encoder=True,
                       classifier_hidden_dim=24,
                       num_tasks=1)

    assert not model.encoder.training
    assert not model.projector.training
    assert isinstance(model.classifier, MultiTaskModel)
    assert model.classifier.training

    model.train()
    assert not model.encoder.training
    assert not model.projector.training
    assert isinstance(model.classifier, MultiTaskModel)
    assert model.classifier.training

    model.eval()
    assert not model.encoder.training
    assert not model.projector.training
    assert isinstance(model.classifier, MultiTaskModel)
    assert not model.classifier.training

    model = ImageModel(img_model_type='resnet18',
                       joint_feature_size=4,
                       num_classes=2,
                       freeze_encoder=False,
                       classifier_hidden_dim=24,
                       num_tasks=1)
    assert model.encoder.training
    assert model.projector.training
    assert model.classifier.training  # type: ignore


def test_image_get_patchwise_projected_embeddings() -> None:
    """
    Checks if the image patch embeddings are correctly computed and projected to the latent space.
    """

    num_classes = 2
    num_tasks = 1
    joint_feature_size = 4
    model = ImageModel(img_model_type='resnet18',
                       joint_feature_size=joint_feature_size,
                       num_classes=num_classes,
                       freeze_encoder=True,
                       classifier_hidden_dim=None,
                       num_tasks=num_tasks)
    model.train()
    with pytest.raises(AssertionError) as ex:
        model.get_patchwise_projected_embeddings(torch.rand(size=(2, 3, 448, 448)), normalize=True)
    assert "This function is only implemented for evaluation mode" in str(ex)
    model.eval()

    batch_size = 2
    image = torch.rand(size=(batch_size, 3, 64, 64))
    encoder_output, _ = model.encoder.forward(image, return_patch_embeddings=True)
    h, w = encoder_output.shape[2:]

    # First check the model output is in the expected shape,
    # since this is used internally by get_patchwise_projected_embeddings
    model_output = model.forward(image)
    assert model_output.projected_patch_embeddings.shape == (batch_size, joint_feature_size, h, w)
    assert model_output.projected_global_embedding.shape == (batch_size, joint_feature_size)
    projected_global_embedding = model_output.projected_global_embedding

    unnormalized_patch_embeddings = model.get_patchwise_projected_embeddings(image, normalize=False)
    # Make sure the projected embeddings returned are the right shape
    assert unnormalized_patch_embeddings.shape == (batch_size, h, w, joint_feature_size)
    result_1 = torch.mean(unnormalized_patch_embeddings, dim=(1, 2))  # B x W x H x D
    result_2 = projected_global_embedding
    assert torch.allclose(result_1, result_2)

    # test normalized version
    normalized_patch_embeddings = model.get_patchwise_projected_embeddings(image, normalize=True)
    assert normalized_patch_embeddings.shape == (batch_size, h, w, joint_feature_size)
    # Make sure the norm is 1 along the embedding dimension
    norm = normalized_patch_embeddings.norm(p=2, dim=-1)
    assert torch.all(torch.abs(norm - 1.0) < 1e-5)


def test_reload_resnet_with_dilation() -> None:
    """
    Tests if the resnet model can be switched from pooling to dilated convolutions.
    """

    replace_stride_with_dilation = [False, False, True]

    # resnet18 does not support dilation
    model_with_dilation = ImageEncoder(img_model_type="resnet18")
    with pytest.raises(NotImplementedError):
        model_with_dilation.reload_encoder_with_dilation(replace_stride_with_dilation)

    # resnet50
    original_model = ImageEncoder(img_model_type="resnet50").eval()
    model_with_dilation = ImageEncoder(img_model_type="resnet50").eval()
    model_with_dilation.reload_encoder_with_dilation(replace_stride_with_dilation)
    assert not model_with_dilation.training

    batch_size = 2
    image = torch.rand(size=(batch_size, 3, 64, 64))

    with torch.no_grad():
        outputs_dilation, _ = model_with_dilation(image, return_patch_embeddings=True)
        outputs_original, _ = original_model(image, return_patch_embeddings=True)
        assert outputs_original.shape[2] * \
            2 == outputs_dilation.shape[2], "The dilation model should return larger feature maps."

    expected_model = resnet50(pretrained=True, replace_stride_with_dilation=replace_stride_with_dilation)

    expected_model.eval()
    with torch.no_grad():
        expected_output = expected_model(image)
        assert torch.allclose(outputs_dilation, expected_output)


@torch.no_grad()
def test_hubconf() -> None:
    """Test that instantiating the image model using the PyTorch Hub is consistent with older methods."""
    image = torch.rand(1, 3, 480, 480)

    github = 'microsoft/hi-ml:main'
    model_hub = torch.hub.load(github, 'biovil_resnet', pretrained=True)
    model_himl = get_biovil_resnet()

    output_hub: ImageModelOutput = model_hub(image)
    output_himl: ImageModelOutput = model_himl(image)

    for field_himl in fields(output_himl):
        value_hub = getattr(output_hub, field_himl.name)
        value_himl = getattr(output_himl, field_himl.name)
        if value_hub is None and value_himl is None:  # for example, class_logits
            continue
        assert torch.allclose(value_hub, value_himl)


def test_restore_training_mode() -> None:
    model = torch.nn.Conv2d(3, 2, 3)
    assert model.training

    with restore_training_mode(model):
        assert model.training
        model.eval()
        assert not model.training
    assert model.training

    model.eval()
    assert not model.training
    with restore_training_mode(model):
        assert not model.training
        model.train()
        assert model.training
    assert not model.training
