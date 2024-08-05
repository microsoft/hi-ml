#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Sequence

import pytest
import torch
from health_multimodal.image.model.encoder import (
    DEFAULT_DILATION_VALUES_FOR_RESNET,
    ImageEncoder,
    MultiImageEncoder,
    restore_training_mode,
)
from health_multimodal.image.model.pretrained import get_imagenet_init_encoder
from health_multimodal.image.model.resnet import resnet50
from health_multimodal.image.model.types import ImageEncoderType


@pytest.mark.parametrize("replace_stride_with_dilation", [None, [False, False, True]])
def test_reload_resnet_with_dilation(replace_stride_with_dilation: Sequence[bool]) -> None:
    """
    Tests if the resnet model can be switched from pooling to dilated convolutions.
    """

    # resnet18 does not support dilation
    model_with_dilation = ImageEncoder(img_encoder_type=ImageEncoderType.RESNET18)
    with pytest.raises(NotImplementedError):
        model_with_dilation.reload_encoder_with_dilation(replace_stride_with_dilation)

    # resnet50
    original_model = get_imagenet_init_encoder().encoder.eval()
    model_with_dilation = get_imagenet_init_encoder().encoder.eval()
    model_with_dilation.reload_encoder_with_dilation(replace_stride_with_dilation)
    assert not model_with_dilation.training

    batch_size = 2
    image = torch.rand(size=(batch_size, 3, 64, 64))

    with torch.no_grad():
        outputs_dilation, _ = model_with_dilation(image, return_patch_embeddings=True)
        outputs_original, _ = original_model(image, return_patch_embeddings=True)
        assert (
            outputs_original.shape[2] * 2 == outputs_dilation.shape[2]
        ), "The dilation model should return larger feature maps."

    resnet50_kwargs = {
        "pretrained": True,
        "replace_stride_with_dilation": (
            replace_stride_with_dilation if replace_stride_with_dilation else DEFAULT_DILATION_VALUES_FOR_RESNET
        ),
    }
    expected_model = resnet50(**resnet50_kwargs)

    expected_model.eval()
    with torch.no_grad():
        expected_output = expected_model(image)
        print(expected_output.shape)
        print(outputs_dilation.shape)
        assert torch.allclose(outputs_dilation, expected_output)


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


def test_multi_image_encoder_forward_pass() -> None:
    encoder = MultiImageEncoder(img_encoder_type=ImageEncoderType.RESNET18_MULTI_IMAGE)
    assert encoder.training

    batch_size = 2
    current_image = torch.rand(size=(batch_size, 3, 448, 448))
    previous_image = torch.rand(size=(batch_size, 3, 448, 448))

    # Multi-image run
    with torch.no_grad():
        patch_emb, global_emb = encoder(
            current_image=current_image, previous_image=previous_image, return_patch_embeddings=True
        )
        # Model output dimension is fixed to 512 (see MultiImageEncoder.__init__())
        assert global_emb.shape == (batch_size, 512)
        assert patch_emb.shape == (batch_size, 512, 14, 14)

    # Single-image run
    with torch.no_grad():
        patch_emb, global_emb = encoder(current_image=current_image, previous_image=None, return_patch_embeddings=True)
        # Model output dimension is fixed to 512 (see MultiImageEncoder.__init__())
        assert global_emb.shape == (batch_size, 512)
        assert patch_emb.shape == (batch_size, 512, 14, 14)
